# -*- coding: utf-8 -*-

import enum
import os
import subprocess
import sys
import contextlib

import tarfile
import zipfile

import nibabel as nib

from os.path import join as pjoin
from hashlib import md5
from shutil import copyfileobj

from urllib.request import urlopen

from dipy.core.gradients import \
    (gradient_table)  # , gradient_table_from_gradient_strength_bvecs)
from dipy.io.gradients import read_bvals_bvecs
# from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Origin, Space
from dipy.io.streamline import load_tractogram
# from dipy.tracking.streamline import Streamlines


# Set a user-writeable file-system location to put files:
if "TRACTODATA_HOME" in os.environ:
    tractodata_home = os.environ["TRACTODATA_HOME"]
else:
    tractodata_home = pjoin(os.path.expanduser("~"), ".tractodata")


TRACTODATA_DATASETS_URL = "https://osf.io/"


class Dataset(enum.Enum):
    FIBERCUP_ANAT = "fibercup_anat"
    FIBERCUP_DWI = "fibercup_dwi"
    FIBERCUP_SYNTH_TRACKING = "fibercup_synth_tracking"
    FIBERCUP_SYNTH_BUNDLING = "fibercup_synth_bundling"
    ISBI2013_ANAT = "isbi2013_anat"
    ISBI2013_DWI = "isbi2013_dwi"
    ISBI2013_TRACTOGRAPHY = "isbi2013_tractography"
    ISMRM2015_ANAT = "ismrm2015_anat"
    ISMRM2015_DWI = "ismrm2015_dwi"
    ISMRM2015_TRACTOGRAPHY = "ismrm2015_tractography"


class FetcherError(Exception):
    pass


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
        "#" * block + "-" * (bar_length - block), progress * 100, size_string)
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
    """Print a message indicating that dataset is already in place.
    """

    msg = "Dataset is already in place. If you want to fetch it again "
    msg += "please first remove the folder {}".format(folder)
    print(msg)


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
        for chunk in iter(lambda: f.read(128*hash_data.block_size), b""):
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
            msg = \
                "The downloaded file\n{}\ndoes not have the expected hash " \
                "value of {}.\nInstead, the hash value was {}.\nThis could " \
                "mean that something is wrong with the file or that the " \
                "upstream file has been updated.\nYou can try downloading " \
                "file again or updating to the newest version of {}".format(
                    filename, stored_hash, computed_hash,
                    __name__.split('.')[0])
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
        print("Creating new folder {}".format(folder))
        os.makedirs(folder)

    if data_size is not None:
        print("Data size is approximately {}".format(data_size))

    all_skip = True
    for f in files:
        url, _hash = files[f]
        fullpath = pjoin(folder, f)
        if os.path.exists(fullpath) and (_get_file_hash(fullpath) == _hash.lower()):  # noqa E501
            continue
        all_skip = False
        print("Downloading {} to {}".format(f, folder))
        _get_file_data(fullpath, url)
        check_hash(fullpath, _hash)
    if all_skip:
        _already_there_msg(folder)
    else:
        print("Files successfully downloaded to {}".format(folder))


def _make_fetcher(name, folder, baseurl, remote_fnames, local_fnames,
                  hash_list=None, doc="", data_size=None, msg=None,
                  unzip=False):
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
        for i, (f, n), in enumerate(zip(remote_fnames, local_fnames)):
            files[n] = (baseurl + f, hash_list[i] if
                        hash_list is not None else None)
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
                    files[f] += (tuple(z.namelist()), )
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
    TRACTODATA_DATASETS_URL + "ptv25/",
    ["download"],
    ["T1w.nii.gz"],
    ["7170d0192fa00b5ef069f8e7c274950c"],
    data_size="543B",
    doc="Download Fiber Cup dataset anatomy data",
    unzip=False
    )

fetch_fibercup_dwi = _make_fetcher(
    "fetch_fibercup_dwi",
    pjoin(tractodata_home, "datasets", "fibercup", "raw", "sub-01", "dwi"),
    TRACTODATA_DATASETS_URL + "5yqvw/",
    ["download"],
    ["dwi.zip"],
    ["f907901563254833c5f2bc90c209b4ae"],
    data_size="0.39MB",
    doc="Download Fiber Cup dataset diffusion data",
    unzip=True
    )

fetch_fibercup_synth_tracking = _make_fetcher(
    "fetch_fibercup_synth_tracking",
    pjoin(
        tractodata_home, "datasets", "fibercup", "derivatives", "tracking",
        "synth", "sub-01", "dwi"),
    TRACTODATA_DATASETS_URL + "kcng7/",
    ["download"],
    ["dwi_space-orig_desc-synth_tractography.trk"],
    ["9b46bbd9381f589037b5b0077c91ed55"],
    data_size="10.35MB",
    doc="Download Fiber Cup dataset synthetic tracking data",
    unzip=False
    )

fetch_fibercup_synth_bundling = _make_fetcher(
    "fetch_fibercup_synth_bundling",
    pjoin(
        tractodata_home, "datasets", "fibercup", "derivatives", "bundling",
        "synth", "sub-01", "dwi"),
    TRACTODATA_DATASETS_URL + "kp74n/",
    ["download"],
    ["dwi_space-orig_desc-synth_subset-bundles_tractography.zip"],
    ["5930a340edc0a2fb4b59e8c21e154759"],
    data_size="8.55MB",
    doc="Download Fiber Cup dataset synthetic bundle data",
    unzip=True
    )

fetch_isbi2013_anat = _make_fetcher(
    "fetch_isbi2013_anat",
    pjoin(tractodata_home, "datasets", "isbi2013", "raw", "sub-01", "anat"),
    TRACTODATA_DATASETS_URL + "datasets/" + "isbi2013/" + "raw/" +
    "sub-01/" + "anat/",
    ["T1w.nii.gz"],
    ["T1w.nii.gz"],
    ["file_SHA",
     "file_SHA"],
    data_size="12KB",
    doc="Download ISBI 2013 HARDI Reconstruction Challenge dataset anatomy data",  # noqa E501
    unzip=False
    )

fetch_isbi2013_dwi = _make_fetcher(
    "fetch_isbi2013_dwi",
    pjoin(tractodata_home, "datasets", "isbi2013", "raw", "sub-01", "dwi"),
    TRACTODATA_DATASETS_URL + "datasets/" + "isbi2013/" + "raw/" +
    "sub-01/" + "dwi/",
    ["dwi.nii.gz", "dwi.bvals", "dwi.bvecs"],
    ["dwi.nii.gz", "dwi.bvals", "dwi.bvecs"],
    ["file1_SHA",
     "file2_SHA",
     "file3_SHA"],
    data_size="12KB",
    doc="Download ISBI 2013 HARDI Reconstruction Challenge diffusion data",
    unzip=True
    )

fetch_isbi2013_tractography = _make_fetcher(
    "fetch_isbi2013_tractography",
    pjoin(
        tractodata_home, "datasets", "isbi2013", "derivatives", "tractography",
        "sub-01", "dwi"),
    TRACTODATA_DATASETS_URL + "datasets/" + "isbi2013/" + "derivatives/" +
    "tractography/" + "sub-01/" + "dwi/",
    ["trk", "trk", "trk"],
    ["trk", "trk", "trk"],
    ["file1_SHA",
     "file2_SHA",
     "file3_SHA"],
    data_size="12KB",
    doc="Download ISBI 2013 HARDI Reconstruction Challenge tractography data",
    unzip=True
    )


fetch_ismrm2015_anat = _make_fetcher(
    "fetch_ismrm2015_anat",
    pjoin(tractodata_home, "datasets", "ismrm2015", "raw", "sub-01", "anat"),
    TRACTODATA_DATASETS_URL + "datasets/" + "ismrm2015/" + "raw/" +
    "sub-01/" + "anat/",
    ["T1w.nii.gz"],
    ["T1w.nii.gz"],
    ["file1_SHA"],
    data_size="12KB",
    doc="Download ISMRM 2015 Tractography Challenge dataset anatomy data",
    unzip=False
    )

fetch_ismrm2015_dwi = _make_fetcher(
    "fetch_ismrm2015_dwi",
    pjoin(tractodata_home, "datasets", "ismrm2015", "raw", "sub-01", "dwi"),
    TRACTODATA_DATASETS_URL + "datasets/" + "ismrm2015/" + "raw/" +
    "sub-01/" + "dwi/",
    ["dwi.nii.gz", "dwi.bvals", "dwi.bvecs"],
    ["dwi.nii.gz", "dwi.bvals", "dwi.bvecs"],
    ["file1_SHA",
     "file2_SHA",
     "file3_SHA"],
    data_size="12KB",
    doc="Download ISMRM 2015 Tractography Challenge dataset diffusion data",
    unzip=True
    )

fetch_ismrm2015_tractography = _make_fetcher(
    "fetch_ismrm2015_tractography",
    pjoin(
        tractodata_home, "datasets", "ismrm2015", "derivatives",
        "tractography", "sub-01", "dwi"),
    TRACTODATA_DATASETS_URL + "datasets/" + "ismrm2015/" + "derivatives/" +
    "tractography/" + "sub-01/" + "dwi/",
    ["CC.trk",
     "Cingulum_left.trk",
     "Cingulum_right.trk",
     "CST_left.trk",
     "CST_right.trk",
     "FPT_left.trk",
     "FPT_right.trk",
     "Fornix.trk",
     "ICP_left.trk",
     "ICP_right.trk",
     "ILF_left.trk",
     "ILF_right.trk",
     "MCP.trk",
     "OR_left.trk",
     "OR_right.trk",
     "POPT_left.trk",
     "POPT_right.trk",
     "SCP_left.trk",
     "SCP_right.trk",
     "SLF_left.trk",
     "SLF_right.trk",
     "UF_left.trk",
     "UF_right.trk",
     "CST.trk",
     "OR.trk"],
    ["CC.trk",
     "Cingulum_left.trk",
     "Cingulum_right.trk",
     "CST_left.trk",
     "CST_right.trk",
     "FPT_left.trk",
     "FPT_right.trk",
     "Fornix.trk",
     "ICP_left.trk",
     "ICP_right.trk",
     "ILF_left.trk",
     "ILF_right.trk",
     "MCP.trk",
     "OR_left.trk",
     "OR_right.trk",
     "POPT_left.trk",
     "POPT_right.trk",
     "SCP_left.trk",
     "SCP_right.trk",
     "SLF_left.trk",
     "SLF_right.trk",
     "UF_left.trk",
     "UF_right.trk",
     "CST.trk",
     "OR.trk"],
    ["file1_SHA",
     "file2_SHA"
     "file3_SHA",
     "file4_SHA",
     "file5_SHA",
     "file6_SHA",
     "file7_SHA",
     "file8_SHA",
     "file9_SHA",
     "file10_SHA",
     "file11_SHA",
     "file12_SHA",
     "file13_SHA",
     "file14_SHA",
     "file15_SHA",
     "file16_SHA",
     "file17_SHA",
     "file18_SHA",
     "file19_SHA",
     "file20_SHA",
     "file21_SHA",
     "file22_SHA",
     "file23_SHA",
     "file24_SHA",
     "file25_SHA"],
    data_size="12KB",
    doc="Download ISMRM 2015 Tractography Challenge tractography data",
    unzip=True
    )

fetch_ismrm2015_qb = _make_fetcher(
    "fetch_ismrm2015_qb",
    pjoin(
        tractodata_home, "datasets", "ismrm2015", "derivatives", "qb",
        "sub-01", "dwi"),
    TRACTODATA_DATASETS_URL + "datasets/" + "ismrm2015/" + "derivatives/" +
    "qb/" + "sub-01/" + "dwi/",
    ["bundles_attributes.json"],
    ["bundles_attributes.json"],
    ["file1_SHA"],
    data_size="12KB",
    doc="Download ISMRM 2015 Tractography Challenge QuickBundles centroid config data",  # noqa E501
    unzip=True
    )

fetch_ismrm2015_submission_res = _make_fetcher(
    "fetch_ismrm2015_submission_res",
    pjoin(
        tractodata_home, "datasets", "ismrm2015", "derivatives", "submissions",
        "sub-01", "dwi"),
    TRACTODATA_DATASETS_URL + "datasets/" + "ismrm2015/" + "derivatives/" +
    "submissions/" + "sub-01/" + "dwi/",
    ["ismrm2015_tractography_challenge_overall_results.csv",
     "ismrm2015_tractography_challenge_submission1-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission2-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission2-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission4-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission4-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission5-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission5-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission5-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission5-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission8-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission8-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-5_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-5_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-6_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-6_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-7_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-7_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-8_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-8_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-9_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-9_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-10_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-10_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-11_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-11_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-12_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-12_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-13_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-13_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-14_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-14_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-15_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-15_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-16_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-16_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-17_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-17_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-18_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-18_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-19_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-19_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission11-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission11-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission11-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission11-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission14-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission14-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission14-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission14-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission14-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission14-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission15-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission15-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission19-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission19-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission19-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission19-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission19-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission19-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-5_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-5_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-6_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-6_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-7_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-7_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-8_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-8_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-9_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-9_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-10_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-10_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-11_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-11_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-12_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-12_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-13_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-13_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-14_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-14_individual_bundle_results.csv"],  # noqa E501
    ["ismrm2015_tractography_challenge_overall_results.csv",
     "ismrm2015_tractography_challenge_submission1-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission1-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission2-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission2-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission3-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission4-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission4-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission5-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission5-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission5-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission5-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission6-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission7-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission8-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission8-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission9-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-5_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-5_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-6_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-6_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-7_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-7_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-8_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-8_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-9_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-9_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-10_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-10_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-11_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-11_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-12_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-12_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-13_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-13_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-14_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-14_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-15_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-15_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-16_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-16_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-17_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-17_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-18_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-18_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-19_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission10-19_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission11-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission11-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission11-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission11-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission12-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission13-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission14-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission14-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission14-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission14-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission14-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission14-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission15-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission15-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission16-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission17-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission18-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission19-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission19-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission19-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission19-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission19-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission19-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-0_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-0_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-1_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-1_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-2_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-2_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-3_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-3_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-4_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-4_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-5_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-5_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-6_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-6_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-7_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-7_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-8_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-8_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-9_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-9_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-10_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-10_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-11_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-11_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-12_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-12_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-13_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-13_individual_bundle_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-14_angular_error_results.csv",  # noqa E501
     "ismrm2015_tractography_challenge_submission20-14_individual_bundle_results.csv"],  # noqa E501
     ["ismrm2015_tractography_challenge_overall_results_SHA",
      "file1-0_angular_error_results_SHA",
      "file1-0_individual_bundle_results_SHA",
      "file1-1_angular_error_results_SHA",
      "file1-1_individual_bundle_results_SHA",
      "file1-2_angular_error_results_SHA",
      "file1-2_individual_bundle_results_SHA",
      "file1-3_angular_error_results_SHA",
      "file1-3_individual_bundle_results_SHA",
      "file1-4_angular_error_results_SHA",
      "file1-4_individual_bundle_results_SHA",
      "file2-0_angular_error_results_SHA",
      "file2-0_individual_bundle_results_SHA",
      "file3-0_angular_error_results_SHA",
      "file3-0_individual_bundle_results_SHA",
      "file3-1_angular_error_results_SHA",
      "file3-1_individual_bundle_results_SHA",
      "file3-2_angular_error_results_SHA",
      "file3-2_individual_bundle_results_SHA",
      "file3-3_angular_error_results_SHA",
      "file3-3_individual_bundle_results_SHA",
      "file3-4_angular_error_results_SHA",
      "file3-4_individual_bundle_results_SHA",
      "file4-0_angular_error_results_SHA",
      "file4-0_individual_bundle_results_SHA",
      "file5-0_angular_error_results_SHA",
      "file5-0_individual_bundle_results_SHA",
      "file5-1_angular_error_results_SHA",
      "file5-1_individual_bundle_results_SHA",
      "file6-0_angular_error_results_SHA",
      "file6-0_individual_bundle_results_SHA",
      "file6-1_angular_error_results_SHA",
      "file6-1_individual_bundle_results_SHA",
      "file6-2_angular_error_results_SHA",
      "file6-2_individual_bundle_results_SHA",
      "file6-3_angular_error_results_SHA",
      "file6-3_individual_bundle_results_SHA",
      "file6-4_angular_error_results_SHA",
      "file6-4_individual_bundle_results_SHA",
      "file7-0_angular_error_results_SHA",
      "file7-0_individual_bundle_results_SHA",
      "file7-1_angular_error_results_SHA",
      "file7-1_individual_bundle_results_SHA",
      "file7-2_angular_error_results_SHA",
      "file7-2_individual_bundle_results_SHA",
      "file7-3_angular_error_results_SHA",
      "file7-3_individual_bundle_results_SHA",
      "file8-0_angular_error_results_SHA",
      "file8-0_individual_bundle_results_SHA",
      "file9-0_angular_error_results_SHA",
      "file9-0_individual_bundle_results_SHA",
      "file9-1_angular_error_results_SHA",
      "file9-1_individual_bundle_results_SHA",
      "file9-2_angular_error_results_SHA",
      "file9-2_individual_bundle_results_SHA",
      "file9-3_angular_error_results_SHA",
      "file9-3_individual_bundle_results_SHA",
      "file9-4_angular_error_results_SHA",
      "file9-4_individual_bundle_results_SHA",
      "file10-0_angular_error_results_SHA",
      "file10-0_individual_bundle_results_SHA",
      "file10-1_angular_error_results_SHA",
      "file10-1_individual_bundle_results_SHA",
      "file10-2_angular_error_results_SHA",
      "file10-2_individual_bundle_results_SHA",
      "file10-3_angular_error_results_SHA",
      "file10-3_individual_bundle_results_SHA",
      "file10-4_angular_error_results_SHA",
      "file10-4_individual_bundle_results_SHA",
      "file10-5_angular_error_results_SHA",
      "file10-5_individual_bundle_results_SHA",
      "file10-6_angular_error_results_SHA",
      "file10-6_individual_bundle_results_SHA",
      "file10-7_angular_error_results_SHA",
      "file10-7_individual_bundle_results_SHA",
      "file10-8_angular_error_results_SHA",
      "file10-8_individual_bundle_results_SHA",
      "file10-9_angular_error_results_SHA",
      "file10-9_individual_bundle_results_SHA",
      "file10-10_angular_error_results_SHA",
      "file10-10_individual_bundle_results_SHA",
      "file10-11_angular_error_results_SHA",
      "file10-11_individual_bundle_results_SHA",
      "file10-12_angular_error_results_SHA",
      "file10-12_individual_bundle_results_SHA",
      "file10-13_angular_error_results_SHA",
      "file10-13_individual_bundle_results_SHA",
      "file10-14_angular_error_results_SHA",
      "file10-14_individual_bundle_results_SHA",
      "file10-15_angular_error_results_SHA",
      "file10-15_individual_bundle_results_SHA",
      "file10-16_angular_error_results_SHA",
      "file10-16_individual_bundle_results_SHA",
      "file10-17_angular_error_results_SHA",
      "file10-17_individual_bundle_results_SHA",
      "file10-18_angular_error_results_SHA",
      "file10-18_individual_bundle_results_SHA",
      "file10-19_angular_error_results_SHA",
      "file10-19_individual_bundle_results_SHA",
      "file11-0_angular_error_results_SHA",
      "file11-0_individual_bundle_results_SHA",
      "file11-1_angular_error_results_SHA",
      "file11-1_individual_bundle_results_SHA",
      "file12-0_angular_error_results_SHA",
      "file12-0_individual_bundle_results_SHA",
      "file12-1_angular_error_results_SHA",
      "file12-1_individual_bundle_results_SHA",
      "file12-2_angular_error_results_SHA",
      "file12-2_individual_bundle_results_SHA",
      "file12-3_angular_error_results_SHA",
      "file12-3_individual_bundle_results_SHA",
      "file13-0_angular_error_results_SHA",
      "file13-0_individual_bundle_results_SHA",
      "file13-1_angular_error_results_SHA",
      "file13-1_individual_bundle_results_SHA",
      "file13-2_angular_error_results_SHA",
      "file13-2_individual_bundle_results_SHA",
      "file13-3_angular_error_results_SHA",
      "file13-3_individual_bundle_results_SHA",
      "file14-0_angular_error_results_SHA",
      "file14-0_individual_bundle_results_SHA",
      "file14-1_angular_error_results_SHA",
      "file14-1_individual_bundle_results_SHA",
      "file14-2_angular_error_results_SHA",
      "file14-2_individual_bundle_results_SHA",
      "file15-0_angular_error_results_SHA",
      "file15-0_individual_bundle_results_SHA",
      "file16-0_angular_error_results_SHA",
      "file16-0_individual_bundle_results_SHA",
      "file16-1_angular_error_results_SHA",
      "file16-1_individual_bundle_results_SHA",
      "file16-2_angular_error_results_SHA",
      "file16-2_individual_bundle_results_SHA",
      "file16-3_angular_error_results_SHA",
      "file16-3_individual_bundle_results_SHA",
      "file16-4_angular_error_results_SHA",
      "file16-4_individual_bundle_results_SHA",
      "file17-0_angular_error_results_SHA",
      "file17-0_individual_bundle_results_SHA",
      "file17-1_angular_error_results_SHA",
      "file17-1_individual_bundle_results_SHA",
      "file17-2_angular_error_results_SHA",
      "file17-2_individual_bundle_results_SHA",
      "file17-3_angular_error_results_SHA",
      "file17-3_individual_bundle_results_SHA",
      "file17-4_angular_error_results_SHA",
      "file17-4_individual_bundle_results_SHA",
      "file18-0_angular_error_results_SHA",
      "file18-0_individual_bundle_results_SHA",
      "file18-1_angular_error_results_SHA",
      "file18-1_individual_bundle_results_SHA",
      "file18-2_angular_error_results_SHA",
      "file18-2_individual_bundle_results_SHA",
      "file18-3_angular_error_results_SHA",
      "file18-3_individual_bundle_results_SHA",
      "file18-4_angular_error_results_SHA",
      "file18-4_individual_bundle_results_SHA",
      "file19-0_angular_error_results_SHA",
      "file19-0_individual_bundle_results_SHA",
      "file19-1_angular_error_results_SHA",
      "file19-1_individual_bundle_results_SHA",
      "file19-2_angular_error_results_SHA",
      "file19-2_individual_bundle_results_SHA",
      "file20-0_angular_error_results_SHA",
      "file20-0_individual_bundle_results_SHA",
      "file20-1_angular_error_results_SHA",
      "file20-1_individual_bundle_results_SHA",
      "file20-2_angular_error_results_SHA",
      "file20-2_individual_bundle_results_SHA",
      "file20-3_angular_error_results_SHA",
      "file20-3_individual_bundle_results_SHA",
      "file20-4_angular_error_results_SHA",
      "file20-4_individual_bundle_results_SHA",
      "file20-5_angular_error_results_SHA",
      "file20-5_individual_bundle_results_SHA",
      "file20-6_angular_error_results_SHA",
      "file20-6_individual_bundle_results_SHA",
      "file20-7_angular_error_results_SHA",
      "file20-7_individual_bundle_results_SHA",
      "file20-8_angular_error_results_SHA",
      "file20-8_individual_bundle_results_SHA",
      "file20-9_angular_error_results_SHA",
      "file20-9_individual_bundle_results_SHA",
      "file20-10_angular_error_results_SHA",
      "file20-10_individual_bundle_results_SHA",
      "file20-11_angular_error_results_SHA",
      "file20-11_individual_bundle_results_SHA",
      "file20-12_angular_error_results_SHA",
      "file20-12_individual_bundle_results_SHA",
      "file20-13_angular_error_results_SHA",
      "file20-13_individual_bundle_results_SHA",
      "file20-14_angular_error_results_SHA",
      "file20-14_individual_bundle_results_SHA"],
    data_size="12KB",
    doc="Download ISMRM 2015 Tractography Challenge submission result data",
    unzip=True
    )


def get_fnames(name):
    """Provide full paths to example or test datasets.

    Parameters
    ----------
    name : Dataset enum
        Dataset name.
    Returns
    -------
    fnames : tuple
        Filenames for dataset.
    """

    print("\nDataset: {}".format(name))

    if name == Dataset.FIBERCUP_ANAT.name:
        files, folder = fetch_fibercup_anat()
        return pjoin(folder, list(files.keys())[0])  # ,"T1w.nii.gz")
    elif name == Dataset.FIBERCUP_DWI.name:
        files, folder = fetch_fibercup_dwi()
        fnames = files['dwi.zip'][2]
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.FIBERCUP_SYNTH_TRACKING.name:
        files, folder = fetch_fibercup_synth_tracking()
        return pjoin(folder, list(files.keys())[0])
    elif name == Dataset.FIBERCUP_SYNTH_BUNDLING.name:
        files, folder = fetch_fibercup_synth_bundling()
        fnames = files[
            'dwi_space-orig_desc-synth_subset-bundles_tractography.zip'][2]
        return sorted([pjoin(folder, f) for f in fnames])
    elif name == Dataset.ISBI2013_ANAT.name:
        files, folder = fetch_isbi2013_anat()
        return pjoin(folder, list(files.keys())[0])  # "T1w.nii.gz")
    elif name == Dataset.ISBI2013_DWI.name:
        files, folder = fetch_isbi2013_dwi()
        fraw = pjoin(folder, list(files.keys())[0])  # "dwi.nii.gz")
        fbval = pjoin(folder, list(files.keys())[1])  # ".bval")
        fbvec = pjoin(folder, list(files.keys())[2])  # "bvec")
        return fraw, fbval, fbvec
    elif name ==  Dataset.ISBI2013_TRACTOGRAPHY.name:
        files, folder = fetch_isbi2013_tractography()
        for fname in list(files.keys()):
            fnames = pjoin(folder, fname)
        return fnames
    elif name == Dataset.ISMRM2015_ANAT.name:
        files, folder = fetch_ismrm2015_anat()
        return pjoin(folder, list(files.keys())[0])  # , "T1w.nii.gz")
    elif name == Dataset.ISMRM2015_DWI.name:
        files, folder = fetch_ismrm2015_dwi()
        fraw = pjoin(folder, list(files.keys())[0])  # "dwi.nii.gz")
        fbval = pjoin(folder, list(files.keys())[1])  # ".bval")
        fbvec = pjoin(folder, list(files.keys())[2])  # "bvec")
        return fraw, fbval, fbvec
    elif name == Dataset.ISMRM2015_TRACTOGRAPHY.name:
        files, folder = fetch_ismrm2015_tractography()
        for fname in list(files.keys()):
            fnames = pjoin(folder, fname)
        return fnames
    else:
        raise ValueError("Unknown dataset.\n"
                         "Provided: {}; Available: {}".
                         format(name, Dataset.__members__.keys()))


def read_fibercup_anat():
    """Load Fiber Cup dataset anatomy data.

    Returns
    -------
    img : obj,
        Nifti1Image.
    """

    fname = get_fnames(Dataset.FIBERCUP_ANAT)
    return nib.load(fname)


def read_fibercup_dwi():
    """Load Fiber Cup dataset diffusion data.

    Returns
    -------
    img : obj,
        Nifti1Image.
    gtab : obj,
        GradientTable.
    """

    fraw, fbval, fbvec = get_fnames(Dataset.FIBERCUP_DWI)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab


def read_fibercup_tractography(bundle_names=None):
    """Load Fiber Cup dataset tractography data.

    Parameters
    ----------
    bundle_names : list
        e.g., ["af", "cst_right", "cc"]. See all the available bundles
        in the ``fibercup`` directory of your``$HOME/.tractodata`` folder.

    Returns
    -------
    bundles : dict
        Dictionary with data of the bundles and the bundles as keys.
    """

    space = Space.RASMM
    origin = Origin.NIFTI

    path = pjoin(
        tractodata_home, "datasets", "fibercup", "derivatives", "tractography",
        "sub-01", "dwi")

    anat_fname = get_fnames(Dataset.FIBERCUP_ANAT)

    bundles = dict()

    for bundle in bundle_names:

        sft = load_tractogram(
            pjoin(path, bundle + ".trk"), anat_fname, to_space=space,
            to_origin=origin, bbox_valid_check=True,
            trk_header_check=True).streamlines

        bundles[bundle] = sft

    return bundles


def read_isbi2013_anat():
    """Load ISBI 2013 HARDI Reconstruction Challenge dataset anatomy data.

    Returns
    -------
    img : obj,
        Nifti1Image.
    """

    fname = get_fnames(Dataset.ISBI2013_ANAT)
    return nib.load(fname)


def read_isbi2013_dwi():
    """Load ISBI 2013 HARDI Reconstruction Challenge dataset diffusion data.

    Returns
    -------
    img : obj,
        Nifti1Image.
    gtab : obj,
        GradientTable.
    """

    fraw, fbval, fbvec = get_fnames(Dataset.ISBI2013_DWI)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab


def read_isbi2013_tractography(bundle_names=None):
    """Load ISBI 2013 HARDI Reconstruction Challenge dataset tractography data.

    Parameters
    ----------
    bundle_names : list
        e.g., ["af", "cst_right", "cc"]. See all the available bundles
        in the ``isbi2013`` directory of your``$HOME/.tractodata`` folder.

    Returns
    -------
    bundles : dict
        Dictionary with data of the bundles and the bundles as keys.
    """

    space = Space.RASMM
    origin = Origin.NIFTI

    path = pjoin(
        tractodata_home, "datasets", "isbi2013", "derivatives", "tractography",
        "sub-01", "dwi")

    anat_fname = get_fnames(Dataset.ISBI2013_ANAT)

    bundles = dict()

    for bundle in bundle_names:

        sft = load_tractogram(
            pjoin(path, bundle + ".trk"), anat_fname, to_space=space,
            to_origin=origin, bbox_valid_check=True,
            trk_header_check=True).streamlines

        bundles[bundle] = sft

    return bundles


def read_ismrm2015_anat():
    """Load ISMRM 2015 Tractography Challenge dataset anatomy data.

    Returns
    -------
    img : obj,
        Nifti1Image.
    gtab : obj,
        GradientTable.
    """

    fname = get_fnames(Dataset.ISMRM2015_ANAT)
    return nib.load(fname)


def read_ismrm2015_dwi():
    """Load ISMRM 2015 Tractography Challenge dataset diffusion data.

    Returns
    -------
    img : obj,
        Nifti1Image.
    gtab : obj,
        GradientTable.
    """

    fraw, fbval, fbvec = get_fnames(Dataset.ISMRM2015_DWI)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab


def read_ismrm2015_tractography(bundle_names=None):
    """Load ISMRM 2015 Tractography Challenge dataset tractography data.

    Parameters
    ----------
    bundle_names : list
        e.g., ["af", "cst_right", "cc"]. See all the available bundles
        in the ``ismrm2015`` directory of your``$HOME/.tractodata`` folder.

    Returns
    -------
    bundles : dict
        Dictionary with data of the bundles and the bundles as keys.
    """

    space = Space.RASMM
    origin = Origin.NIFTI

    path = pjoin(
        tractodata_home, "datasets", "ismrm2015", "derivatives",
        "tractography", "sub-01", "dwi")

    anat_fname = get_fnames(Dataset.ISMRM2015_ANAT)

    bundles = dict()

    for bundle in bundle_names:

        sft = load_tractogram(
            pjoin(path, bundle + ".trk"), anat_fname, to_space=space,
            to_origin=origin, bbox_valid_check=True,
            trk_header_check=True).streamlines

        bundles[bundle] = sft

    return bundles
