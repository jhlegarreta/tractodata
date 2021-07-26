# -*- coding: utf-8 -*-

import contextlib
import enum
import itertools
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from hashlib import md5
from os.path import join as pjoin
from shutil import copyfileobj
from urllib.request import urlopen

import nibabel as nib
import pandas as pd
import trimeshpy
from dipy.core.gradients import (
    gradient_table,  # , gradient_table_from_gradient_strength_bvecs)
)
from dipy.io.gradients import read_bvals_bvecs

# from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Origin, Space
from dipy.io.streamline import load_tractogram

from tractodata.io.utils import (
    Label,
    filter_filenames_on_value,
    get_label_value_from_filename,
    get_longest_common_subseq,
)

# from dipy.tracking.streamline import Streamlines


# Set a user-writeable file-system location to put files:
if "TRACTODATA_HOME" in os.environ:
    tractodata_home = os.environ["TRACTODATA_HOME"]
else:
    tractodata_home = pjoin(os.path.expanduser("~"), ".tractodata")


TRACTODATA_DATASETS_URL = "https://osf.io/"

key_separator = ","

ismrm2015_submission_angular_error_filename_label = "angular_error"
ismrm2015_submission_individual_bundle_filename_label = "individual_bundle"
ismrm2015_submission_overall_filename_label = "overall"

ismrm2015_submission_data_bundle_label = "Bundle"
ismrm2015_submission_data_roi_label = "ROI"
ismrm2015_submission_data_submission_id_label = "Submission ID"


class Dataset(enum.Enum):
    FIBERCUP_ANAT = "fibercup_anat"
    FIBERCUP_DWI = "fibercup_dwi"
    FIBERCUP_TISSUE_MAPS = "fibercup_tissue_maps"
    FIBERCUP_SYNTH_TRACKING = "fibercup_synth_tracking"
    FIBERCUP_SYNTH_BUNDLING = "fibercup_synth_bundling"
    FIBERCUP_SYNTH_BUNDLE_CENTROIDS = "fibercup_synth_bundle_centroids"
    FIBERCUP_BUNDLE_MASKS = "fibercup_bundle_masks"
    FIBERCUP_BUNDLE_ENDPOINT_MASKS = "fibercup_bundle_endpoint_masks"
    FIBERCUP_DIFFUSION_PEAKS = "fibercup_diffusion_peaks"
    FIBERCUP_LOCAL_PROB_TRACKING = "fibercup_local_prob_tracking"
    FIBERCUP_LOCAL_PROB_BUNDLING = "fibercup_local_prob_bundling"
    FIBERCUP_TRACKING_EVALUATION_CONFIG = "fibercup_tracking_evaluation_config"
    HCP_TR_ANAT = "hcp_tr_anat"
    HCP_TR_DTI_MAPS = "hcp_tr_dti_maps"
    HCP_TR_PVE_MAPS = "hcp_tr_pve_maps"
    HCP_TR_EXCLUDE_INCLUDE_MAPS = "hcp_tr_exclude_include_maps"
    HCP_TR_SURFACES = "hcp_tr_surfaces"
    HCP_TR_PFT_TRACKING = "hcp_tr_pft_tracking"
    # ISBI2013_ANAT = "isbi2013_anat"
    # ISBI2013_DWI = "isbi2013_dwi"
    # ISBI2013_TRACTOGRAPHY = "isbi2013_tractography"
    ISMRM2015_ANAT = "ismrm2015_anat"
    ISMRM2015_DWI = "ismrm2015_dwi"
    ISMRM2015_TISSUE_MAPS = "ismrm2015_tissue_maps"
    ISMRM2015_SURFACES = "ismrm2015_surfaces"
    ISMRM2015_DTI_MAPS = "ismrm2015_dti_maps"
    ISMRM2015_SYNTH_TRACKING = "ismrm2015_synth_tracking"
    ISMRM2015_SYNTH_BUNDLING = "ismrm2015_synth_bundling"
    ISMRM2015_BUNDLE_MASKS = "ismrm2015_bundle_masks"
    ISMRM2015_BUNDLE_ENDPOINT_MASKS = "ismrm2015_bundle_endpoint_masks"
    ISMRM2015_CHALLENGE_SUBMISSION = "ismrm2015_challenge_submission"
    ISMRM2015_TRACKING_EVALUATION_CONFIG = (
        "ismrm2015_tracking_evaluation_config"  # noqa E501
    )
    MNI2009CNONLINSYMM_ANAT = "mni2009cnonlinsymm_anat"
    MNI2009CNONLINSYMM_SURFACES = "mni2009cnonlinsymm_surfaces"


class FetcherError(Exception):
    pass


class DatasetError(Exception):
    pass


def _build_bundle_key(bundle_name, hemisphere=None):
    """Build the key for the given bundle: append the hemisphere (if any) with
    a separator.

    Parameters
    ----------
    bundle_name : string
        Bundle name.
    hemisphere : string, optional
        Hemisphere.

    Returns
    ----------
    key : string
        Key value.
    """

    key = bundle_name
    if hemisphere:
        key += key_separator + hemisphere

    return key


def _build_bundle_endpoint_key(bundle_name, endpoint, hemisphere=None):
    """Build the key for the given bundle endpoint: append the hemisphere (if
    any) and the endpoint with a separator.

    Parameters
    ----------
    bundle_name : string
        Bundle name.
    endpoint : string
        Endpoint name.
    hemisphere : string, optional
        Hemisphere.

    Returns
    ----------
    key : string
        Key value.
    """

    key = bundle_name
    if hemisphere:
        key += key_separator + hemisphere

    key += key_separator + endpoint

    return key


def _build_surface_key(surface_type, hemisphere=None):
    """Build the key for the given surface: append the hemisphere (if any) with
    a separator.

    Parameters
    ----------
    surface_type : string
        Surface type.
    hemisphere : string, optional
        Hemisphere.

    Returns
    -------
    key : string
        Key value.
    """

    key = surface_type
    if hemisphere:
        key += key_separator + hemisphere

    return key


def _check_known_dataset(name):
    """Raise a DatasetError if the dataset is unknown.

    Parameters
    ----------
    name : string
        Dataset name.
    """

    if name not in Dataset.__members__.keys():
        raise DatasetError(_unknown_dataset_msg(name))


def _exclude_dataset_use_permission_files(fnames, permission_fname):
    """Exclude dataset use permission files from the data filenames.

    Parameters
    ----------
    fnames : list
        Filenames.

    Returns
    -------
    key : string
        Key value.
    """

    return [f for f in fnames if permission_fname not in f]


def update_progressbar(progress, total_length):
    """Show progressbar.

    Takes a number between 0 and 1 to indicate progress from 0 to 100%.
    """

    # Try to set the bar_length according to the console size
    # noinspection PyBroadException
    try:
        columns = subprocess.Popen("tput cols", "r").read()
        bar_length = int(columns) - 46
        if bar_length < 1:
            bar_length = 20
    except Exception:
        # Default value if determination of console size fails
        bar_length = 20
    block = int(round(bar_length * progress))
    size_string = "{0:.2f} MB".format(float(total_length) / (1024 * 1024))
    text = "\rDownload Progress: [{0}] {1:.2f}%  of {2}".format(
        "#" * block + "-" * (bar_length - block), progress * 100, size_string
    )
    sys.stdout.write(text)
    sys.stdout.flush()


def copyfileobj_withprogress(fsrc, fdst, total_length, length=16 * 1024):

    copied = 0
    while True:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
        copied += len(buf)
        progress = float(copied) / float(total_length)
        update_progressbar(progress, total_length)


def _already_there_msg(folder):
    """Print a message indicating that dataset is already in place."""

    msg = "Dataset is already in place.\nIf you want to fetch it again, "
    msg += "please first remove the file at issue in folder\n{}".format(folder)
    print(msg)


def _unknown_dataset_msg(name):
    """Build a message indicating that dataset is not known.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    msg : string
        Message.
    """

    msg = "Unknown dataset.\nProvided: {}; Available: {}".format(
        name, Dataset.__members__.keys()
    )
    return msg


def _get_file_hash(filename):
    """Generate an MD5 hash for the entire file in blocks of 128.

    Parameters
    ----------
    filename : str
        The path to the file whose MD5 hash is to be generated.

    Returns
    -------
    hash256_data : str
        The computed MD5 hash from the input file.
    """

    hash_data = md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * hash_data.block_size), b""):
            hash_data.update(chunk)
    return hash_data.hexdigest()


def check_hash(filename, stored_hash=None):
    """Check that the hash of the given filename equals the stored one.

    Parameters
    ----------
    filename : str
        The path to the file whose hash is to be compared.
    stored_hash : str, optional
        Used to verify the generated hash.
        Default: None, checking is skipped.
    """

    if stored_hash is not None:
        computed_hash = _get_file_hash(filename)
        if stored_hash.lower() != computed_hash:
            msg = (
                "The downloaded file\n{}\ndoes not have the expected hash "
                "value of {}.\nInstead, the hash value was {}.\nThis could "
                "mean that something is wrong with the file or that the "
                "upstream file has been updated.\nYou can try downloading "
                "file again or updating to the newest version of {}".format(
                    filename,
                    stored_hash,
                    computed_hash,
                    __name__.split(".")[0],
                )
            )
            raise FetcherError(msg)


def _get_file_data(fname, url):

    with contextlib.closing(urlopen(url)) as opener:
        try:
            response_size = opener.headers["content-length"]
        except KeyError:
            response_size = None

        with open(fname, "wb") as data:
            if response_size is None:
                copyfileobj(opener, data)
            else:
                copyfileobj_withprogress(opener, data, response_size)


def fetch_data(files, folder, data_size=None):
    """Download files to folder and checks their hashes.

    Parameters
    ----------
    files : dictionary
        For each file in ``files`` the value should be (url, hash). The file
        will be downloaded from url if the file does not already exist or if
        the file exists but the hash does not match.
    folder : str
        The directory where to save the file, the directory will be created if
        it does not already exist.
    data_size : str, optional
        A string describing the size of the data (e.g. "91 MB") to be logged to
        the screen. Default does not produce any information about data size.

    Raises
    ------
    FetcherError
        Raises if the hash of the file does not match the expected value. The
        downloaded file is not deleted when this error is raised.
    """

    if not os.path.exists(folder):
        print("Creating new folder\n{}".format(folder))
        os.makedirs(folder)

    if data_size is not None:
        print("Data size is approximately {}".format(data_size))

    all_skip = True
    for f in files:
        url, _hash = files[f]
        fullpath = pjoin(folder, f)
        if os.path.exists(fullpath) and (
            _get_file_hash(fullpath) == _hash.lower()
        ):  # noqa E501
            continue
        all_skip = False
        print("Downloading\n{}\nto\n{}".format(f, folder))
        _get_file_data(fullpath, url)
        check_hash(fullpath, _hash)
    if all_skip:
        _already_there_msg(folder)
    else:
        print("\nFiles successfully downloaded to\n{}".format(folder))


def _make_fetcher(
    name,
    folder,
    baseurl,
    remote_fnames,
    local_fnames,
    hash_list=None,
    doc="",
    data_size=None,
    msg=None,
    unzip=False,
):
    """Create a new fetcher.

    Parameters
    ----------
    name : str
        The name of the fetcher function.
    folder : str
        The full path to the folder in which the files would be placed locally.
        Typically, this is something like "pjoin(tractodata_home, "foo")"
    baseurl : str
        The URL from which this fetcher reads files.
    remote_fnames : list of strings
        The names of the files in the baseurl location.
    local_fnames : list of strings
        The names of the files to be saved on the local filesystem.
    hash_list : list of strings, optional
        The hash values of the files. Used to verify the content of the files.
        Default: None, skipping checking hash.
    doc : str, optional.
        Documentation of the fetcher.
    data_size : str, optional.
        If provided, is sent as a message to the user before downloading
        starts.
    msg : str, optional
        A message to print to screen when fetching takes place. Default (None)
        is to print nothing.
    unzip : bool, optional
        Whether to unzip the file(s) after downloading them. Supports zip, gz,
        and tar.gz files.

    Returns
    -------
    fetcher : function
        A function that, when called, fetches data according to the designated
        inputs

    """

    def fetcher():
        files = {}
        for (
            i,
            (f, n),
        ) in enumerate(zip(remote_fnames, local_fnames)):
            files[n] = (
                baseurl + f,
                hash_list[i] if hash_list is not None else None,
            )
        fetch_data(files, folder, data_size)

        if msg is not None:
            print(msg)
        if unzip:
            for f in local_fnames:
                split_ext = os.path.splitext(f)
                if split_ext[-1] == ".gz" or split_ext[-1] == ".bz2":
                    if os.path.splitext(split_ext[0])[-1] == ".tar":
                        ar = tarfile.open(pjoin(folder, f))
                        ar.extractall(path=folder)
                        ar.close()
                    else:
                        raise ValueError("File extension is not recognized")
                elif split_ext[-1] == ".zip":
                    z = zipfile.ZipFile(pjoin(folder, f), "r")
                    files[f] += (tuple(z.namelist()),)
                    z.extractall(folder)
                    z.close()
                else:
                    raise ValueError("File extension is not recognized")

        return files, folder

    fetcher.__name__ = name
    fetcher.__doc__ = doc
    return fetcher


fetch_fibercup_anat = _make_fetcher(
    "fetch_fibercup_anat",
    pjoin(tractodata_home, "datasets", "fibercup", "raw", "sub-01", "anat"),
    TRACTODATA_DATASETS_URL + "2xmgw/",
    ["download"],
    ["sub01-T1w.nii.gz"],
    ["7170d0192fa00b5ef069f8e7c274950c"],
    data_size="543B",
    doc="Download Fiber Cup dataset anatomy data",
    unzip=False,
)

fetch_fibercup_dwi = _make_fetcher(
    "fetch_fibercup_dwi",
    pjoin(tractodata_home, "datasets", "fibercup", "raw", "sub-01", "dwi"),
    TRACTODATA_DATASETS_URL + "br4ds/",
    ["download"],
    ["sub01-dwi.zip"],
    ["705396981f1bcda51de12098db968390"],
    data_size="0.39MB",
    doc="Download Fiber Cup dataset diffusion data",
    unzip=True,
)

fetch_fibercup_tissue_maps = _make_fetcher(
    "fetch_fibercup_tissue_maps",
    pjoin(
        tractodata_home,
        "datasets",
        "fibercup",
        "derivatives",
        "segmentation",
        "synth",
        "sub-01",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "z8qea//",
    ["download"],
    ["sub01-T1w_space-orig_dseg.zip"],
    ["98e09f049676fe35c593baa33d1d0524"],
    data_size="808B",
    doc="Download Fiber Cup dataset tissue maps",
    unzip=True,
)

fetch_fibercup_synth_tracking = _make_fetcher(
    "fetch_fibercup_synth_tracking",
    pjoin(
        tractodata_home,
        "datasets",
        "fibercup",
        "derivatives",
        "tracking",
        "synth",
        "sub-01",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "ug7d6/",
    ["download"],
    ["sub01-dwi_space-orig_desc-synth_tractography.trk"],
    ["9b46bbd9381f589037b5b0077c91ed55"],
    data_size="10.35MB",
    doc="Download Fiber Cup dataset synthetic tracking data",
    unzip=False,
)

fetch_fibercup_synth_bundling = _make_fetcher(
    "fetch_fibercup_synth_bundling",
    pjoin(
        tractodata_home,
        "datasets",
        "fibercup",
        "derivatives",
        "bundling",
        "synth",
        "sub-01",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "h84w6/",
    ["download"],
    ["sub01-dwi_space-orig_desc-synth_subset-bundles_tractography.zip"],
    ["60589568bc13d4093af5bb282d78e9ff"],
    data_size="8.55MB",
    doc="Download Fiber Cup dataset synthetic bundling data",
    unzip=True,
)

fetch_fibercup_synth_bundle_centroids = _make_fetcher(
    "fetch_fibercup_synth_bundle_centroids",
    pjoin(
        tractodata_home,
        "datasets",
        "fibercup",
        "derivatives",
        "centroids",
        "quickbundles",
        "sub-01",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "7eaxv/",
    ["download"],
    ["sub01-dwi_space-orig_desc-synth_subset-bundles_centroid.zip"],
    ["c60f4206dd0dfc1f1e1a4f282935eee4"],
    data_size="20.7KB",
    doc="Download Fiber Cup dataset synthetic QuickBundles bundle centroid data",  # noqa E501
    unzip=True,
)

fetch_fibercup_bundle_masks = _make_fetcher(
    "fetch_fibercup_bundle_masks",
    pjoin(
        tractodata_home,
        "datasets",
        "fibercup",
        "derivatives",
        "bundling",
        "synth",
        "sub-01",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "r5f9q/",
    ["download"],
    ["sub01-T1w_space-orig_desc-synth_subset-bundles_tractography.zip"],
    ["e46d1e634e0c5b6a062d2da03edf7c0a"],
    data_size="0.5MB",
    doc="Download Fiber Cup dataset synthetic bundle masks",
    unzip=True,
)

fetch_fibercup_bundle_endpoint_masks = _make_fetcher(
    "fetch_fibercup_bundle_endpoint_masks",
    pjoin(
        tractodata_home,
        "datasets",
        "fibercup",
        "derivatives",
        "connectivity",
        "synth",
        "sub-01",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "y7b2r/",
    ["download"],
    [
        "sub01-T1w_space-orig_desc-synth_subset-bundles_part-endpoints_tractography.zip"
    ],  # noqa E501
    ["ad8efab1c4743aa83df242c77b61c102"],
    data_size="6.6KB",
    doc="Download Fiber Cup dataset synthetic bundle endpoint masks",
    unzip=True,
)

fetch_fibercup_diffusion_peaks = _make_fetcher(
    "fetch_fibercup_diffusion_peaks",
    pjoin(
        tractodata_home,
        "datasets",
        "fibercup",
        "derivatives",
        "diffusion_peaks",
        "dipy_csd",
        "sub-01",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "ezqa3/",
    ["download"],
    ["sub01-dwi_space-orig_model-CSD_PEAKS.nii.gz"],
    ["1914dc2c9c26efaf181058f5b4f9480c"],
    data_size="48KB",
    doc="Download Fiber Cup dataset diffusion model peaks",
    unzip=False,
)

fetch_fibercup_local_prob_tracking = _make_fetcher(
    "fetch_fibercup_local_prob_tracking",
    pjoin(
        tractodata_home,
        "datasets",
        "fibercup",
        "derivatives",
        "tracking",
        "dipy_local_prob",
        "sub-01",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "4zs6w/",
    ["download"],
    ["sub01-dwi_space-orig_desc-PROB_tractography.trk"],
    ["0136b3accd6314e684426eb4e21b99b7"],
    data_size="16MB",
    doc="Download Fiber Cup dataset local probabilistic tracking data",
    unzip=False,
)

fetch_fibercup_local_prob_bundling = _make_fetcher(
    "fetch_fibercup_local_prob_bundling",
    pjoin(
        tractodata_home,
        "datasets",
        "fibercup",
        "derivatives",
        "bundling",
        "quickbundles",
        "sub-01",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "9hr2e/",
    ["download"],
    ["sub01-dwi_space-orig_desc-PROB_subset-bundles_tractography.zip"],
    ["399af174b025b03dcada6632cb759591"],
    data_size="2.1MB",
    doc="Download Fiber Cup dataset local probabilistic bundling data",
    unzip=True,
)

fetch_fibercup_tracking_evaluation_config = _make_fetcher(
    "fetch_fibercup_tracking_evaluation_config",
    pjoin(
        tractodata_home,
        "datasets",
        "fibercup",
        "derivatives",
        "scoring",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "r3h54/",
    ["download"],
    ["tracking_evaluation_config.json"],
    ["6399cb13a9600acee1ad8fe69437a5af"],
    data_size="917B",
    doc="Download Fiber Cup dataset tracking evaluation config file",
    unzip=False,
)

fetch_hcp_tr_anat = _make_fetcher(
    "fetch_hcp_tr_anat",
    pjoin(
        tractodata_home,
        "datasets",
        "hcp_tr",
        "derivatives",
        "structural",
        "tractoflow_fsl",
        "sub-103818_re",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "8xedb/",
    ["download"],
    ["sub103818_re-T1w_space-MNI152NLin2009cSym.nii.gz"],
    ["3e0adbf95d5c48519bb00f1492f52e39"],
    data_size="3.6MB",
    doc="Download HCP Test-Retest subject retest dataset anatomy data",
    unzip=False,
)

fetch_hcp_tr_dti_maps = _make_fetcher(
    "fetch_hcp_tr_dti_maps",
    pjoin(
        tractodata_home,
        "datasets",
        "hcp_tr",
        "derivatives",
        "diffusion",
        "tractoflow_fsl",
        "sub-103818_re",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "3zmsn/",
    ["download"],
    ["sub103818_re-dwi_space-MNI152NLin2009cSym_model-DTI.zip"],
    ["1a52dd87c4a9519435be2d81ee1e9d76"],
    data_size="2.8MB",
    doc="Download HCP Test-Retest subject retest dataset DTI maps",
    unzip=True,
)

fetch_hcp_tr_pve_maps = _make_fetcher(
    "fetch_hcp_tr_pve_maps",
    pjoin(
        tractodata_home,
        "datasets",
        "hcp_tr",
        "derivatives",
        "segmentation",
        "fast",
        "sub-103818_re",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "pabwx/",
    ["download"],
    ["sub103818_re-T1w_space-MNI152NLin2009cSym_probseg.zip"],
    ["5cefd06349f18a2f05a3fea1992a7eca"],
    data_size="1.6MB",
    doc="Download HCP Test-Retest subject retest dataset PVE map data",
    unzip=True,
)

fetch_hcp_tr_exclude_include_maps = _make_fetcher(
    "fetch_hcp_tr_exclude_include_maps",
    pjoin(
        tractodata_home,
        "datasets",
        "hcp_tr",
        "derivatives",
        "structural",
        "tractoflow_fsl",
        "sub-103818_re",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "u8sbp/",
    ["download"],
    ["sub103818_re-T1w_space-MNI152NLin2009cSym_exclude_include.zip"],
    ["574692ec2baf7fec8d2381c5be48a408"],
    data_size="1.3MB",
    doc="Download HCP Test-Retest subject retest dataset exclude/include map data",
    unzip=True,
)

fetch_hcp_tr_surfaces = _make_fetcher(
    "fetch_hcp_tr_surfaces",
    pjoin(
        tractodata_home,
        "datasets",
        "hcp_tr",
        "derivatives",
        "surface",
        "set_nf_civet",
        "sub-103818_re",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "n89q2/",
    ["download"],
    ["sub103818_re-T1w_space-MNI152NLin2009cSym_LPS.surf.zip"],
    ["0fa063e5b648a7b64d11ae0948573043"],
    data_size="3.9MB",
    doc="Download HCP Test-Retest subject retest dataset surface data",
    unzip=True,
)

fetch_hcp_tr_pft_tracking = _make_fetcher(
    "fetch_hcp_tr_pft_tracking",
    pjoin(
        tractodata_home,
        "datasets",
        "hcp_tr",
        "derivatives",
        "tracking",
        "tractoflow_fsl",
        "sub-103818_re",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "xwc8b/",
    ["download"],
    ["sub103818_re-dwi_space-MNI152NLin2009cSym_desc-PFT_tractography.trk"],
    ["dbc59743f56e6372018359613a6ff262"],
    data_size="4.2MB",
    doc="Download HCP Test-Retest subject retest dataset PFT tracking data",
    unzip=False,
)

fetch_isbi2013_anat = _make_fetcher(
    "fetch_isbi2013_anat",
    pjoin(tractodata_home, "datasets", "isbi2013", "raw", "sub-01", "anat"),
    TRACTODATA_DATASETS_URL
    + "datasets/"
    + "isbi2013/"
    + "raw/"
    + "sub-01/"
    + "anat/",
    ["T1w.nii.gz"],
    ["T1w.nii.gz"],
    ["file_SHA", "file_SHA"],
    data_size="12KB",
    doc="Download ISBI 2013 HARDI Reconstruction Challenge dataset anatomy data",  # noqa E501
    unzip=False,
)

fetch_isbi2013_dwi = _make_fetcher(
    "fetch_isbi2013_dwi",
    pjoin(tractodata_home, "datasets", "isbi2013", "raw", "sub-01", "dwi"),
    TRACTODATA_DATASETS_URL
    + "datasets/"
    + "isbi2013/"
    + "raw/"
    + "sub-01/"
    + "dwi/",
    ["dwi.nii.gz", "dwi.bvals", "dwi.bvecs"],
    ["dwi.nii.gz", "dwi.bvals", "dwi.bvecs"],
    ["file1_SHA", "file2_SHA", "file3_SHA"],
    data_size="12KB",
    doc="Download ISBI 2013 HARDI Reconstruction Challenge diffusion data",
    unzip=True,
)

fetch_isbi2013_tractography = _make_fetcher(
    "fetch_isbi2013_tractography",
    pjoin(
        tractodata_home,
        "datasets",
        "isbi2013",
        "derivatives",
        "tractography",
        "sub-01",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL
    + "datasets/"
    + "isbi2013/"
    + "derivatives/"
    + "tractography/"
    + "sub-01/"
    + "dwi/",
    ["trk", "trk", "trk"],
    ["trk", "trk", "trk"],
    ["file1_SHA", "file2_SHA", "file3_SHA"],
    data_size="12KB",
    doc="Download ISBI 2013 HARDI Reconstruction Challenge tractography data",
    unzip=True,
)

fetch_ismrm2015_anat = _make_fetcher(
    "fetch_ismrm2015_anat",
    pjoin(tractodata_home, "datasets", "ismrm2015", "raw", "sub-01", "anat"),
    TRACTODATA_DATASETS_URL + "gdvch/",
    ["download"],
    ["sub01-T1w.nii.gz"],
    ["65af72af2824abce0243cb09555e3a6c"],
    data_size="7.3MB",
    doc="Download ISMRM 2015 Tractography Challenge dataset anatomy data",
    unzip=False,
)

fetch_ismrm2015_dwi = _make_fetcher(
    "fetch_ismrm2015_dwi",
    pjoin(tractodata_home, "datasets", "ismrm2015", "raw", "sub-01", "dwi"),
    TRACTODATA_DATASETS_URL + "4s9ev/",
    ["download"],
    ["sub01-dwi.zip"],
    ["3f228979ca1960f25aa9abc14dc708b8"],
    data_size="7.1MB",
    doc="Download ISMRM 2015 Tractography Challenge dataset diffusion data",
    unzip=True,
)

fetch_ismrm2015_tissue_maps = _make_fetcher(
    "fetch_ismrm2015_tissue_maps",
    pjoin(
        tractodata_home,
        "datasets",
        "ismrm2015",
        "derivatives",
        "segmentation",
        "synth",
        "sub-01",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "b3z54//",
    ["download"],
    ["sub01-T1w_space-orig_dseg.zip"],
    ["04c1518480d79d603b126e2c436c697a"],
    data_size="205.9KB",
    doc="Download ISMRM 2015 Tractography Challenge dataset tissue maps",
    unzip=True,
)

fetch_ismrm2015_surfaces = _make_fetcher(
    "fetch_ismrm2015_surfaces",
    pjoin(
        tractodata_home,
        "datasets",
        "ismrm2015",
        "derivatives",
        "surface",
        "fastsurfer",
        "sub-01",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "yb6d2/",
    ["download"],
    ["sub01-T1w_space-orig_pial.surf.zip"],
    ["33ea5dcd1e863eb4dc8c3063bf3d89fd"],
    data_size="3.7MB",
    doc="Download ISMRM 2015 Tractography Challenge dataset surface data",
    unzip=True,
)

fetch_ismrm2015_dti_maps = _make_fetcher(
    "fetch_ismrm2015_dti_maps",
    pjoin(
        tractodata_home,
        "datasets",
        "ismrm2015",
        "derivatives",
        "diffusion",
        "scilpy",
        "sub-01",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "wnfh2/",
    ["download"],
    ["sub01-dwi_space-orig_desc-WLS_model-DTI.zip"],
    ["ea30620705f123ca6dffcfce9330d0ac"],
    data_size="5.5MB",
    doc="Download ISMRM 2015 Tractography Challenge dataset DTI maps",
    unzip=True,
)

fetch_ismrm2015_synth_tracking = _make_fetcher(
    "fetch_ismrm2015_synth_tracking",
    pjoin(
        tractodata_home,
        "datasets",
        "ismrm2015",
        "derivatives",
        "tracking",
        "synth",
        "sub-01",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "nxmr8/",
    ["download"],
    ["sub01-dwi_space-orig_desc-synth_tractography.trk"],
    ["2a72eeb2949285176344eca31f0b3a39"],
    data_size="235.8MB",
    doc="Download ISMRM 2015 Tractography Challenge dataset synthetic tracking data",  # noqa E501
    unzip=False,
)

fetch_ismrm2015_synth_bundling = _make_fetcher(
    "fetch_ismrm2015_synth_bundling",
    pjoin(
        tractodata_home,
        "datasets",
        "ismrm2015",
        "derivatives",
        "bundling",
        "synth",
        "sub-01",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "5bzaf/",
    ["download"],
    ["sub01-dwi_space-orig_desc-synth_subset-bundles_tractography.zip"],
    ["69daad08e5093fd3eff9a2fbf26777bc"],
    data_size="217.6MB",
    doc="Download ISMRM 2015 Tractography Challenge dataset synthetic bundling data",  # noqa E501
    unzip=True,
)

fetch_ismrm2015_bundle_masks = _make_fetcher(
    "fetch_ismrm2015_bundle_masks",
    pjoin(
        tractodata_home,
        "datasets",
        "ismrm2015",
        "derivatives",
        "bundling",
        "synth",
        "sub-01",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "qy2ap/",
    ["download"],
    ["sub01-T1w_space-orig_desc-synth_subset-bundles_tractography.zip"],
    ["56cd19ba6b57875e582d5d704ec0312f"],
    data_size="543.3KB",
    doc="Download ISMRM 2015 Tractography Challenge dataset synthetic bundle masks",  # noqa E501
    unzip=True,
)

fetch_ismrm2015_bundle_endpoint_masks = _make_fetcher(
    "fetch_ismrm2015_bundle_endpoint_masks",
    pjoin(
        tractodata_home,
        "datasets",
        "ismrm2015",
        "derivatives",
        "connectivity",
        "synth",
        "sub-01",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "24yqs/",
    ["download"],
    [
        "sub01-T1w_space-orig_desc-synth_subset-bundles_part-endpoints_tractography.zip"
    ],  # noqa E501
    ["30d14a729cb100aca6386230cef45284"],
    data_size="203.3KB",
    doc="Download ISMRM 2015 Tractography Challenge dataset synthetic bundle endpoint masks",  # noqa E501
    unzip=True,
)

fetch_ismrm2015_submission_res = _make_fetcher(
    "fetch_ismrm2015_submission_res",
    pjoin(
        tractodata_home,
        "datasets",
        "ismrm2015",
        "derivatives",
        "submission",
        "synth",
        "sub-02",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "t4m9a/",
    ["download"],
    ["sub02-dwi_space-orig_desc-synth_submission_results_tractography.zip"],
    ["6f27a599ac4977a5a91f9111c1726665"],
    data_size="116.8KB",
    doc="Download ISMRM 2015 Tractography Challenge submission result data",
    unzip=True,
)

fetch_ismrm2015_tracking_evaluation_config = _make_fetcher(
    "fetch_ismrm2015_tracking_evaluation_config",
    pjoin(
        tractodata_home,
        "datasets",
        "ismrm2015",
        "derivatives",
        "scoring",
        "dwi",
    ),
    TRACTODATA_DATASETS_URL + "wbdyr/",
    ["download"],
    ["tracking_evaluation_config.json"],
    ["164489da0dc4fb069212543c669ba284"],
    data_size="1.4KB",
    doc="Download ISMRM 2015 Tractography Challenge dataset tracking evaluation config file",  # noqa E501
    unzip=False,
)

fetch_mni2009cnonlinsymm_anat = _make_fetcher(
    "fetch_mni2009cnonlinsymm_anat",
    pjoin(
        tractodata_home,
        "datasets",
        "mni",
        "derivatives",
        "atlas",
        "icbm152_2009c_nonlinsymm",
        "sub-01",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "4hqzj/",
    ["download"],
    ["sub01-T1w.zip"],
    ["0e3455597421e8fb14321d236bae45c9"],
    data_size="4.2MB",
    doc="Download MNI ICBM 2009c Nonlinear Symmetric 1×1x1mm template dataset "
    "brain-masked anatomy data",
    unzip=True,
)

fetch_mni2009cnonlinsymm_surfaces = _make_fetcher(
    "fetch_mni2009cnonlinsymm_surfaces",
    pjoin(
        tractodata_home,
        "datasets",
        "mni",
        "derivatives",
        "surface",
        "fastsurfer",
        "sub-01",
        "anat",
    ),
    TRACTODATA_DATASETS_URL + "4dfv7/",
    ["download"],
    ["sub01-T1w_space-orig_pial.surf.zip"],
    ["b36a14f78ff006a6b881414d45b1111c"],
    data_size="8.1MB",
    doc="Download MNI ICBM 2009c Nonlinear Symmetric 1×1x1mm template dataset "
    "surface data",
    unzip=True,
)


def get_fnames(name):
    """Provide full paths to example or test datasets.

    Parameters
    ----------
    name : string
        Dataset name.
    Returns
    -------
    fnames : string or list
        Filenames for dataset.
    """

    print("\nDataset: {}".format(name))

    if name == Dataset.FIBERCUP_ANAT.name:
        files, folder = fetch_fibercup_anat()
        return pjoin(folder, list(files.keys())[0])
    elif name == Dataset.FIBERCUP_DWI.name:
        files, folder = fetch_fibercup_dwi()
        fnames = files["sub01-dwi.zip"][2]
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.FIBERCUP_TISSUE_MAPS.name:
        files, folder = fetch_fibercup_tissue_maps()
        fnames = files["sub01-T1w_space-orig_dseg.zip"][2]
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.FIBERCUP_SYNTH_TRACKING.name:
        files, folder = fetch_fibercup_synth_tracking()
        return pjoin(folder, list(files.keys())[0])
    elif name == Dataset.FIBERCUP_SYNTH_BUNDLING.name:
        files, folder = fetch_fibercup_synth_bundling()
        fnames = files[
            "sub01-dwi_space-orig_desc-synth_subset-bundles_tractography.zip"
        ][
            2
        ]  # noqa E501
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.FIBERCUP_SYNTH_BUNDLE_CENTROIDS.name:
        files, folder = fetch_fibercup_synth_bundle_centroids()
        fnames = files[
            "sub01-dwi_space-orig_desc-synth_subset-bundles_centroid.zip"
        ][
            2
        ]  # noqa E501
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.FIBERCUP_BUNDLE_MASKS.name:
        files, folder = fetch_fibercup_bundle_masks()
        fnames = files[
            "sub01-T1w_space-orig_desc-synth_subset-bundles_tractography.zip"
        ][
            2
        ]  # noqa E501
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.FIBERCUP_BUNDLE_ENDPOINT_MASKS.name:
        files, folder = fetch_fibercup_bundle_endpoint_masks()
        fnames = files[
            "sub01-T1w_space-orig_desc-synth_subset-bundles_part-endpoints_tractography.zip"
        ][
            2
        ]  # noqa E501
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.FIBERCUP_DIFFUSION_PEAKS.name:
        files, folder = fetch_fibercup_diffusion_peaks()
        return pjoin(folder, list(files.keys())[0])
    elif name == Dataset.FIBERCUP_LOCAL_PROB_TRACKING.name:
        files, folder = fetch_fibercup_local_prob_tracking()
        return pjoin(folder, list(files.keys())[0])
    elif name == Dataset.FIBERCUP_LOCAL_PROB_BUNDLING.name:
        files, folder = fetch_fibercup_local_prob_bundling()
        fnames = files[
            "sub01-dwi_space-orig_desc-PROB_subset-bundles_tractography.zip"
        ][
            2
        ]  # noqa E501
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.FIBERCUP_TRACKING_EVALUATION_CONFIG.name:
        files, folder = fetch_fibercup_tracking_evaluation_config()
        return pjoin(folder, list(files.keys())[0])
    elif name == Dataset.HCP_TR_ANAT.name:
        files, folder = fetch_hcp_tr_anat()
        return pjoin(folder, list(files.keys())[0])
    elif name == Dataset.HCP_TR_DTI_MAPS.name:
        files, folder = fetch_hcp_tr_dti_maps()
        fnames = files[
            "sub103818_re-dwi_space-MNI152NLin2009cSym_model-DTI.zip"
        ][2]
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.HCP_TR_EXCLUDE_INCLUDE_MAPS.name:
        files, folder = fetch_hcp_tr_exclude_include_maps()
        fnames = files[
            "sub103818_re-T1w_space-MNI152NLin2009cSym_exclude_include.zip"
        ][2]
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.HCP_TR_PVE_MAPS.name:
        files, folder = fetch_hcp_tr_pve_maps()
        fnames = files[
            "sub103818_re-T1w_space-MNI152NLin2009cSym_probseg.zip"
        ][2]
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.HCP_TR_SURFACES.name:
        files, folder = fetch_hcp_tr_surfaces()
        fnames = files[
            "sub103818_re-T1w_space-MNI152NLin2009cSym_LPS.surf.zip"
        ][
            2
        ]  # noqa E501
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.HCP_TR_PFT_TRACKING.name:
        files, folder = fetch_hcp_tr_pft_tracking()
        return pjoin(folder, list(files.keys())[0])
    # elif name == Dataset.ISBI2013_ANAT.name:
    #   files, folder = fetch_isbi2013_anat()
    #   return pjoin(folder, list(files.keys())[0])  # "T1w.nii.gz")
    # elif name == Dataset.ISBI2013_DWI.name:
    #   files, folder = fetch_isbi2013_dwi()
    #   fraw = pjoin(folder, list(files.keys())[0])  # "dwi.nii.gz")
    #   fbval = pjoin(folder, list(files.keys())[1])  # ".bval")
    #   fbvec = pjoin(folder, list(files.keys())[2])  # "bvec")
    #   return fraw, fbval, fbvec
    # elif name == Dataset.ISBI2013_TRACTOGRAPHY.name:
    #   files, folder = fetch_isbi2013_tractography()
    #   for fname in list(files.keys()):
    #       fnames = pjoin(folder, fname)
    #   return fnames
    elif name == Dataset.ISMRM2015_ANAT.name:
        files, folder = fetch_ismrm2015_anat()
        return pjoin(folder, list(files.keys())[0])
    elif name == Dataset.ISMRM2015_DTI_MAPS.name:
        files, folder = fetch_ismrm2015_dti_maps()
        fnames = files["sub01-dwi_space-orig_desc-WLS_model-DTI.zip"][2]
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.ISMRM2015_DWI.name:
        files, folder = fetch_ismrm2015_dwi()
        fnames = files["sub01-dwi.zip"][2]
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.ISMRM2015_TISSUE_MAPS.name:
        files, folder = fetch_ismrm2015_tissue_maps()
        fnames = files["sub01-T1w_space-orig_dseg.zip"][2]
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.ISMRM2015_SURFACES.name:
        files, folder = fetch_ismrm2015_surfaces()
        fnames = files["sub01-T1w_space-orig_pial.surf.zip"][2]
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.ISMRM2015_SYNTH_TRACKING.name:
        files, folder = fetch_ismrm2015_synth_tracking()
        return pjoin(folder, list(files.keys())[0])
    elif name == Dataset.ISMRM2015_SYNTH_BUNDLING.name:
        files, folder = fetch_ismrm2015_synth_bundling()
        fnames = files[
            "sub01-dwi_space-orig_desc-synth_subset-bundles_tractography.zip"
        ][
            2
        ]  # noqa E501
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.ISMRM2015_BUNDLE_MASKS.name:
        files, folder = fetch_ismrm2015_bundle_masks()
        fnames = files[
            "sub01-T1w_space-orig_desc-synth_subset-bundles_tractography.zip"
        ][
            2
        ]  # noqa E501
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.ISMRM2015_BUNDLE_ENDPOINT_MASKS.name:
        files, folder = fetch_ismrm2015_bundle_endpoint_masks()
        fnames = files[
            "sub01-T1w_space-orig_desc-synth_subset-bundles_part-endpoints_tractography.zip"
        ][
            2
        ]  # noqa E501
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.ISMRM2015_CHALLENGE_SUBMISSION.name:
        files, folder = fetch_ismrm2015_submission_res()
        fnames = files[
            "sub02-dwi_space-orig_desc-synth_submission_results_tractography.zip"
        ][
            2
        ]  # noqa E501
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.ISMRM2015_TRACKING_EVALUATION_CONFIG.name:
        files, folder = fetch_ismrm2015_tracking_evaluation_config()
        return pjoin(folder, list(files.keys())[0])
    elif name == Dataset.MNI2009CNONLINSYMM_ANAT.name:
        files, folder = fetch_mni2009cnonlinsymm_anat()
        fnames = files["sub01-T1w.zip"][2]
        # Exclude the COPYING file
        fnames = _exclude_dataset_use_permission_files(fnames, "COPYING")
        return pjoin(folder, fnames[0])
    elif name == Dataset.MNI2009CNONLINSYMM_SURFACES.name:
        files, folder = fetch_mni2009cnonlinsymm_surfaces()
        fnames = files["sub01-T1w_space-orig_pial.surf.zip"][2]
        # Exclude the COPYING file
        fnames = _exclude_dataset_use_permission_files(fnames, "COPYING")
        return sorted([pjoin(folder, f) for f in fnames])
    else:
        raise DatasetError(_unknown_dataset_msg(name))


def list_bundles_in_dataset(name):
    """List dataset bundle names.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    bundles : list
        Bundle names.
    """

    _check_known_dataset(name)

    fnames = get_fnames(name)

    bundles = []

    for fname in fnames:
        _bundle = get_label_value_from_filename(fname, Label.BUNDLE)
        hemisphere = get_label_value_from_filename(fname, Label.HEMISPHERE)

        bundle = _build_bundle_key(_bundle, hemisphere=hemisphere)

        bundles.append(bundle)

    return bundles


def list_bundle_endpoint_masks_in_dataset(name):
    """List dataset bundle endpoint_names.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    bundle_endpoints : list
        Bundle endpoint names.
    """

    _check_known_dataset(name)

    fnames = get_fnames(name)

    bundle_endpoints = []

    for fname in fnames:
        _bundle = get_label_value_from_filename(fname, Label.BUNDLE)
        hemisphere = get_label_value_from_filename(fname, Label.HEMISPHERE)
        endpoint = get_label_value_from_filename(fname, Label.ENDPOINT)

        bundle_endpoint = _build_bundle_endpoint_key(
            _bundle, endpoint, hemisphere=hemisphere
        )

        bundle_endpoints.append(bundle_endpoint)

    return bundle_endpoints


def list_dti_maps_in_dataset(name):
    """List dataset DTI map names.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    dti_map_names : list
        DTI map names.
    """

    _check_known_dataset(name)

    fnames = get_fnames(name)

    dti_map_names = []

    for fname in fnames:
        dti_map_name = get_label_value_from_filename(fname, Label.DTI)
        dti_map_names.append(dti_map_name)

    return dti_map_names


def list_exclude_include_maps_in_dataset(name):
    """List dataset exclude/include map names.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    exclude_include_names : list
        Exclude/include map names.
    """

    _check_known_dataset(name)

    fnames = get_fnames(name)

    exclude_include_names = []

    for fname in fnames:
        exclude_include_name = get_label_value_from_filename(
            fname, Label.EXCLUDEINCLUDE
        )
        exclude_include_names.append(exclude_include_name)

    return exclude_include_names


def list_tissue_maps_in_dataset(name):
    """List dataset tissue map names.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    tissue_names : list
        Tissue map names.
    """

    _check_known_dataset(name)

    fnames = get_fnames(name)

    tissue_names = []

    for fname in fnames:
        tissue_name = get_label_value_from_filename(fname, Label.TISSUE)
        tissue_names.append(tissue_name)

    return tissue_names


def list_surfaces_in_dataset(name):
    """List dataset surface names.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    surface_names : list
        Surface names.
    """

    _check_known_dataset(name)

    fnames = get_fnames(name)

    surface_names = []

    has_period = True

    for fname in fnames:
        surface_type = get_label_value_from_filename(
            fname, Label.SURFACE, has_period
        )
        hemisphere = get_label_value_from_filename(
            fname, Label.HEMISPHERE, has_period
        )

        surface_name = _build_surface_key(surface_type, hemisphere=hemisphere)

        surface_names.append(surface_name)

    return surface_names


def read_dataset_anat(name):
    """Load dataset anatomy data.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    img : Nifti1Image
        Anatomy image.
    """

    _check_known_dataset(name)

    fname = get_fnames(name)

    return nib.load(fname)


def read_dataset_dwi(name):
    """Load dataset diffusion data.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    img : Nifti1Image
        Diffusion image.
    gtab : GradientTable
        Diffusion encoding gradient information.
    """

    _check_known_dataset(name)

    bval_fname, bvec_fname, dwi_fname = get_fnames(name)

    bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    img = nib.load(dwi_fname)

    return img, gtab


def read_dataset_dti_maps(name, map_name=None):
    """Load dataset DTI maps.

    Parameters
    ----------
    name : string
        Dataset name.
    map_name : list, optional
        e.g., ["FA"]. See all the available DTI maps
        in the appropriate directory of your ``$HOME/.tractodata`` folder. If
        `None`, all will be loaded.

    Returns
    -------
    Nifti1Image DTI maps.
    """

    _check_known_dataset(name)

    fnames = get_fnames(name)

    dti_maps = dict()

    fnames_shortlist = fnames

    if map_name:
        fnames_shortlist = filter_filenames_on_value(
            fnames, Label.DTI, map_name
        )

    for fname in fnames_shortlist:
        _map_name = get_label_value_from_filename(fname, Label.DTI)

        dti_maps[_map_name] = nib.load(fname)

    return dti_maps


def read_dataset_exclude_include_maps(name, exclude_include_name=None):
    """Load dataset tissue maps.

    Parameters
    ----------
    name : string
        Dataset name.
    exclude_include_name : list, optional
        e.g., ["EXCLUDE", "INCLUDE", "INTERFACE"]. See all the available
        exclude/include names in the appropriate directory of your
        ``$HOME/.tractodata`` folder. If `None`, all will be loaded.

    Returns
    -------
    Nifti1Image exclude/include maps.
    """

    _check_known_dataset(name)

    fnames = get_fnames(name)

    exclude_include_maps = dict()

    fnames_shortlist = fnames

    if exclude_include_name:
        fnames_shortlist = filter_filenames_on_value(
            fnames, Label.EXCLUDEINCLUDE, exclude_include_name
        )

    for fname in fnames_shortlist:
        _exclude_include_name = get_label_value_from_filename(
            fname, Label.EXCLUDEINCLUDE
        )

        exclude_include_maps[_exclude_include_name] = nib.load(fname)

    return exclude_include_maps


def read_dataset_tissue_maps(name, tissue_name=None):
    """Load dataset tissue maps.

    Parameters
    ----------
    name : string
        Dataset name.
    tissue_name : list, optional
        e.g., ["CSF", "GM", "WM"]. See all the available tissues
        in the appropriate directory of your ``$HOME/.tractodata`` folder. If
        `None`, all will be loaded.

    Returns
    -------
    Nifti1Image tissue maps.
    """

    _check_known_dataset(name)

    fnames = get_fnames(name)

    tissue_maps = dict()

    fnames_shortlist = fnames

    if tissue_name:
        fnames_shortlist = filter_filenames_on_value(
            fnames, Label.TISSUE, tissue_name
        )

    for fname in fnames_shortlist:
        _tissue_name = get_label_value_from_filename(fname, Label.TISSUE)

        tissue_maps[_tissue_name] = nib.load(fname)

    return tissue_maps


def read_dataset_surfaces(
    name, surface_type=None, hemisphere_name=None, as_polydata=False
):
    """Load dataset surfaces.

    Parameters
    ----------
    name : string
        Dataset name.
    surface_type : string, optional
        e.g., ["pial"] for the gray matter/pial matter border. If `None` all
        will be loaded.
    hemisphere_name : string, optional
        e.g., ["L", "R"] for left or right hemispheres. If `None` all will be
        loaded.
    as_polydata : bool, optional
        `True` if surfaces are to be loaded as VTK PolyData objects;
         `trimeshpy.TriMesh_Vtk` objects are returned otherwise.

    Returns
    -------
    Surface meshes.
    """

    _check_known_dataset(name)

    fnames = get_fnames(name)

    surfaces = dict()

    fnames_shortlist = fnames

    if hemisphere_name:
        fnames_shortlist = filter_filenames_on_value(
            fnames_shortlist, Label.HEMISPHERE, hemisphere_name
        )

    if surface_type:
        fnames_shortlist = filter_filenames_on_value(
            fnames_shortlist, Label.SURFACE, surface_type
        )

    has_period = True

    for fname in fnames_shortlist:
        _surface_type = get_label_value_from_filename(
            fname, Label.SURFACE, has_period=has_period
        )
        _hemisphere_name = get_label_value_from_filename(
            fname, Label.HEMISPHERE, has_period=has_period
        )

        key = _build_surface_key(_surface_type, hemisphere=_hemisphere_name)

        if as_polydata:
            surfaces[key] = trimeshpy.vtk_util.load_polydata(fname)
        else:
            surfaces[key] = trimeshpy.TriMesh_Vtk(
                fname, None, assert_args=False
            )

    return surfaces


def read_dataset_tracking(
    anat_name, tracking_name, space=Space.RASMM, origin=Origin.NIFTI
):
    """Load dataset tracking data.

    Parameters
    ----------
    anat_name : string
        Anatomy dataset name.
    tracking_name : string
        Tracking dataset name.
    origin : Origin, optional
        Origin for the returned data.
    space : Space, optional
        Space for the returned data.

    Returns
    -------
    sft : StatefulTractogram
        Tractogram.
    """

    _check_known_dataset(anat_name)
    _check_known_dataset(tracking_name)

    anat_fname = get_fnames(anat_name)

    sft_fname = get_fnames(tracking_name)

    sft = load_tractogram(
        sft_fname,
        anat_fname,
        to_space=space,
        to_origin=origin,
        bbox_valid_check=True,
        trk_header_check=True,
    )

    return sft


def read_dataset_bundling(
    anat_name,
    bundling_name,
    bundle_name=None,
    hemisphere_name=None,
    space=Space.RASMM,
    origin=Origin.NIFTI,
):
    """Load dataset bundling data.

    Parameters
    ----------
    anat_name : string
        Anatomy dataset name.
    bundling_name : string
        Bundling dataset name.
    bundle_name : list, optional
        e.g., ["bundle1", "bundle2", "bundle3"]. See all the available bundles
        in the appropriate directory of your ``$HOME/.tractodata`` folder. If
        `None`, all will be loaded.
    hemisphere_name : string, optional
        e.g., ["L", "R"] for left or right hemispheres. If `None` all will be
        loaded.
    origin : Origin, optional
        Origin for the returned data.
    space : Space, optional
        Space for the returned data.

    Returns
    -------
    bundles : dict
        Dictionary with data of the bundles and the bundles as keys.
    """

    _check_known_dataset(anat_name)
    _check_known_dataset(bundling_name)

    anat_fname = get_fnames(anat_name)

    fnames = get_fnames(bundling_name)

    bundles = dict()

    fnames_shortlist = fnames

    if bundle_name:
        fnames_shortlist = filter_filenames_on_value(
            fnames, Label.BUNDLE, bundle_name
        )

    if hemisphere_name:
        fnames_shortlist = filter_filenames_on_value(
            fnames_shortlist, Label.HEMISPHERE, hemisphere_name
        )

    for fname in fnames_shortlist:
        _bundle_name = get_label_value_from_filename(fname, Label.BUNDLE)
        _hemisphere_name = get_label_value_from_filename(
            fname, Label.HEMISPHERE
        )

        key = _build_bundle_key(_bundle_name, hemisphere=_hemisphere_name)

        bundles[key] = load_tractogram(
            fname,
            anat_fname,
            to_space=space,
            to_origin=origin,
            bbox_valid_check=True,
            trk_header_check=True,
        )

    return bundles


def read_dataset_bundle_masks(name, bundle_name=None, hemisphere_name=None):
    """Load dataset bundle masks.

    Parameters
    ----------
    name : string
        Dataset name.
    bundle_name : list, optional
        e.g., ["bundle1", "bundle2", "bundle3"]. See all the available bundles
        in the appropriate directory of your ``$HOME/.tractodata`` folder.
    hemisphere_name : string, optional
        e.g., ["L", "R"] for left or right hemispheres.

    Returns
    -------
    bundles : dict
        Dictionary with masks of the bundles and the bundles as keys.
    """

    _check_known_dataset(name)

    fnames = get_fnames(name)

    bundle_masks = dict()

    fnames_shortlist = fnames

    if bundle_name:
        fnames_shortlist = filter_filenames_on_value(
            fnames, Label.BUNDLE, bundle_name
        )

    if hemisphere_name:
        fnames_shortlist = filter_filenames_on_value(
            fnames_shortlist, Label.HEMISPHERE, hemisphere_name
        )

    for fname in fnames_shortlist:
        _bundle_name = get_label_value_from_filename(fname, Label.BUNDLE)
        _hemisphere_name = get_label_value_from_filename(
            fname, Label.HEMISPHERE
        )

        key = _build_bundle_key(_bundle_name, hemisphere=_hemisphere_name)

        bundle_masks[key] = nib.load(fname)

    return bundle_masks


def read_dataset_bundle_endpoint_masks(
    name, bundle_name=None, hemisphere_name=None, endpoint_name=None
):
    """Load dataset bundle endpoint masks.

    Parameters
    ----------
    name : string
        Dataset name.
    bundle_name : list, optional
        e.g., ["bundle1", "bundle2", "bundle3"]. See all the available bundles
        in the ``fibercup`` directory of your ``$HOME/.tractodata`` folder.
    hemisphere_name : string, optional
        e.g., ["L", "R"] for left or right hemispheres.
    endpoint_name : string, optional
        e.g., ["head", "tail"].

    Returns
    -------
    bundles : dict
        Dictionary with endpoint masks of the bundles and the bundles as keys.
    """

    _check_known_dataset(name)

    fnames = get_fnames(name)

    bundle_endpoint_masks = dict()

    fnames_shortlist = fnames

    if bundle_name:
        fnames_shortlist = filter_filenames_on_value(
            fnames, Label.BUNDLE, bundle_name
        )

    if hemisphere_name:
        fnames_shortlist = filter_filenames_on_value(
            fnames_shortlist, Label.HEMISPHERE, hemisphere_name
        )

    if endpoint_name:
        fnames_shortlist = filter_filenames_on_value(
            fnames_shortlist, Label.ENDPOINT, endpoint_name
        )

    for fname in fnames_shortlist:
        _bundle_name = get_label_value_from_filename(fname, Label.BUNDLE)
        _hemisphere_name = get_label_value_from_filename(
            fname, Label.HEMISPHERE
        )
        _endpoint_name = get_label_value_from_filename(fname, Label.ENDPOINT)

        key = _build_bundle_endpoint_key(
            _bundle_name, _endpoint_name, hemisphere=_hemisphere_name
        )

        bundle_endpoint_masks[key] = nib.load(fname)

    return bundle_endpoint_masks


def read_dataset_diffusion_peaks(name):
    """Load dataset diffusion peak data.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    img : Nifti1Image
        Diffusion peaks image.
    """

    _check_known_dataset(name)

    fname = get_fnames(name)

    return nib.load(fname)


def read_dataset_tracking_evaluation_config(name):
    """Load dataset tracking evaluation configuration data.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    config : dict
        Tracking evaluation configuration data.
    """

    _check_known_dataset(name)

    fname = get_fnames(name)

    with open(fname, "r") as f:
        return json.load(f)


def _get_ismrm2015_submission_id_from_filenames(fnames):

    label = []
    for f in fnames:
        label.append(os.path.splitext(os.path.basename(f))[0])

    # Assume that the filenames have two common substrings, and that the
    # submission id is embedded in between
    common_subseq = get_longest_common_subseq(label)
    label = list(map(lambda st: str.replace(st, common_subseq, ""), label))
    common_subseq = get_longest_common_subseq(label)
    label = list(map(lambda st: str.replace(st, common_subseq, ""), label))

    return label


def _classify_ismrm2015_submissions_results_files():
    """Classify the ISMRM 2015 Tractography Challenge submission result files.

    Returns
    -------
    overall_scores_fname : string
        Overall performance filename.
    angular_error_score_fnames : list
        Angular performance data filenames
    bundle_score_fnames : list
         Bundle performance filenames.
    """

    fnames = get_fnames(Dataset.ISMRM2015_CHALLENGE_SUBMISSION.name)

    overall_scores_fname = [
        f
        for f in fnames
        if ismrm2015_submission_overall_filename_label
        in os.path.splitext(os.path.basename(f))[0]
    ][0]

    angular_error_score_fnames = [
        f
        for f in fnames
        if ismrm2015_submission_angular_error_filename_label
        in os.path.splitext(os.path.basename(f))[0]
    ]
    bundle_score_fnames = [
        f
        for f in fnames
        if ismrm2015_submission_individual_bundle_filename_label
        in os.path.splitext(os.path.basename(f))[0]
    ]

    return (
        overall_scores_fname,
        angular_error_score_fnames,
        bundle_score_fnames,
    )


def read_ismrm2015_submissions_overall_performance_data(score=None):
    """Read ISMRM 2015 Tractography Challenge submission overall performance
    data.

    Parameters
    ----------
    score : list
        Scores whose data is to be retrieved. If `None` all score data will be
        read.

    Returns
    -------
    df : pd.DataFrame
        The overall performance data.
    """

    # Get the relevant file
    fname, _, _ = _classify_ismrm2015_submissions_results_files()

    # Keep the submission id column
    usecols = None
    if score:
        usecols = [ismrm2015_submission_data_submission_id_label] + score

    df = pd.read_csv(fname, usecols=usecols)
    df.set_index(ismrm2015_submission_data_submission_id_label, inplace=True)

    return df


def read_ismrm2015_submissions_angular_performance_data(score=None, roi=None):
    """Read ISMRM 2015 Tractography Challenge submission angular performance
    data.

    Parameters
    ----------
    score : list
        Scores whose data is to be retrieved. If `None`, all score data will be
        read.
    roi : list
        ROIs whose data is to be retrieved. If `None`, all ROI data will be
        read.

    Returns
    -------
    df : pd.DataFrame
        The angular performance data.
    """

    # Get the relevant files
    _, fnames, _ = _classify_ismrm2015_submissions_results_files()

    scores = []
    register_count = []
    for f in fnames:
        # Keep the ROI column
        usecols = None
        if score:
            usecols = [ismrm2015_submission_data_roi_label] + score

        df = pd.read_csv(f, usecols=usecols)
        register_count.append(len(df.index))
        scores.append(df)

    # Create the df with all data
    df = pd.concat(scores, ignore_index=True)

    # Get the submission id to use it as an index together with the bundle
    # names
    submission_id = _get_ismrm2015_submission_id_from_filenames(fnames)
    submission_id = [
        [_id] * count for _id, count in zip(submission_id, register_count)
    ]
    submission_id = list(itertools.chain.from_iterable(submission_id))
    df[ismrm2015_submission_data_submission_id_label] = submission_id

    df.set_index(
        [
            ismrm2015_submission_data_submission_id_label,
            ismrm2015_submission_data_roi_label,
        ],
        inplace=True,
    )

    # Filter the df if ROIs were given
    if roi:
        df = df[df.index.isin(roi, level=1)]

    return df


def read_ismrm2015_submissions_bundle_performance_data(
    score=None, bundle_name=None
):
    """Read ISMRM 2015 Tractography Challenge submission bundle performance
    data.

    Parameters
    ----------
    score : list
        Scores whose data is to be retrieved. If `None`, all score data will be
        read.
    bundle_name : list
        Bundle names whose data is to be retrieved. If `None`, all bundle data
        will be read.

    Returns
    -------
    df : pd.DataFrame
        The bundle performance data.
    """

    # Get the relevant files
    _, _, fnames = _classify_ismrm2015_submissions_results_files()

    scores = []
    register_count = []
    for f in fnames:
        # Keep the bundle column
        usecols = None
        if score:
            usecols = [ismrm2015_submission_data_bundle_label] + score

        df = pd.read_csv(f, usecols=usecols)
        register_count.append(len(df.index))
        scores.append(df)

    # Create the df with all data
    df = pd.concat(scores, ignore_index=True)

    # Get the submission id to use it as an index together with the bundle
    # names
    submission_id = _get_ismrm2015_submission_id_from_filenames(fnames)
    submission_id = [
        [_id] * count for _id, count in zip(submission_id, register_count)
    ]
    submission_id = list(itertools.chain.from_iterable(submission_id))
    df[ismrm2015_submission_data_submission_id_label] = submission_id

    df.set_index(
        [
            ismrm2015_submission_data_submission_id_label,
            ismrm2015_submission_data_bundle_label,
        ],
        inplace=True,
    )

    # Filter the df if bundle names were given
    if bundle_name:
        df = df[df.index.isin(bundle_name, level=1)]

    return df
