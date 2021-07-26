#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import os
import os.path as op
import tempfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from importlib import reload
from os.path import join as pjoin
from threading import Thread
from urllib.request import pathname2url

import nibabel as nib
import numpy as np
import numpy.testing as npt
from dipy.io.streamline import StatefulTractogram
from nibabel.tmpdirs import TemporaryDirectory
from trimeshpy import vtk_util as vtk_u

import tractodata.io.fetcher as fetcher
from tractodata.data import TEST_FILES
from tractodata.io.fetcher import TRACTODATA_DATASETS_URL, Dataset
from tractodata.io.utils import (
    DTIMap,
    Endpoint,
    ExcludeIncludeMap,
    Hemisphere,
    Surface,
    Tissue,
)

fibercup_bundles = [
    "bundle1",
    "bundle2",
    "bundle3",
    "bundle4",
    "bundle5",
    "bundle6",
    "bundle7",
]
fibercup_tissues = [Tissue.WM.value]
ismrm2015_association_bundles = [
    "Cing",
    "FPT",
    "ICP",
    "ILF",
    "OR",
    "POPT",
    "SCP",
    "SLF",
    "UF",
]
ismrm2015_projection_bundles = ["CST"]
ismrm2015_commissural_bundles = ["CC", "CA", "CP", "Fornix", "MCP"]
ismrm2015_dti_maps = [DTIMap.FA.value]
ismrm2015_tissues = [Tissue.WM.value]
ismrm2015_surfaces = [Surface.PIAL.value]

hcp_tr_dti_maps = [DTIMap.FA.value]
hcp_tr_exclude_include_maps = [
    ExcludeIncludeMap.EXCLUDE.value,
    ExcludeIncludeMap.INCLUDE.value,
    ExcludeIncludeMap.INTERFACE.value,
]
hcp_tr_pve_maps = [Tissue.CSF.value, Tissue.GM.value, Tissue.WM.value]

tracking_config_file_necessary_keys = ["cluster_threshold"]


def _build_fibercup_bundle_endpoints():

    return list(
        itertools.product(
            fibercup_bundles, [Endpoint.HEAD.value, Endpoint.TAIL.value]
        )
    )


def _build_ismrm2015_bundles():

    assoc_val = list(
        itertools.product(
            ismrm2015_association_bundles,
            [Hemisphere.LEFT.value, Hemisphere.RIGHT.value],
        )
    )

    proj_val = itertools.product(
        ismrm2015_projection_bundles,
        [Hemisphere.LEFT.value, Hemisphere.RIGHT.value],
    )

    return list(
        itertools.chain.from_iterable(
            [assoc_val, ismrm2015_commissural_bundles, proj_val]
        )
    )


def _build_ismrm2015_bundle_endpoints():

    assoc_val = list(
        itertools.product(
            ismrm2015_association_bundles,
            [Hemisphere.LEFT.value, Hemisphere.RIGHT.value],
            [Endpoint.HEAD.value, Endpoint.TAIL.value],
        )
    )

    commiss_val = list(
        itertools.product(
            ismrm2015_commissural_bundles,
            [Endpoint.HEAD.value, Endpoint.TAIL.value],
        )
    )

    proj_val = list(
        itertools.product(
            ismrm2015_projection_bundles,
            [Hemisphere.LEFT.value, Hemisphere.RIGHT.value],
            [Endpoint.HEAD.value, Endpoint.TAIL.value],
        )
    )

    return list(
        itertools.chain.from_iterable([assoc_val, commiss_val, proj_val])
    )


def _build_ismrm2015_surfaces():

    return list(
        itertools.product(
            ismrm2015_surfaces, [Hemisphere.LEFT.value, Hemisphere.RIGHT.value]
        )
    )


def _check_fibercup_img(img):

    npt.assert_equal(img.__class__.__name__, nib.Nifti1Image.__name__)
    npt.assert_equal(img.get_fdata().dtype, np.float64)
    npt.assert_equal(img.get_fdata().shape, (64, 64, 3))


def _check_hcp_tr_img(img):

    npt.assert_equal(img.__class__.__name__, nib.Nifti1Image.__name__)
    npt.assert_equal(img.get_fdata().dtype, np.float64)
    npt.assert_equal(img.get_fdata().shape, (105, 138, 111))


def _check_ismrm2015_img(img):

    npt.assert_equal(img.__class__.__name__, nib.Nifti1Image.__name__)
    npt.assert_equal(img.get_fdata().dtype, np.float64)
    npt.assert_equal(img.get_fdata().shape, (180, 216, 180))


def _check_mni2009cnonlinsymm_img(img):

    npt.assert_equal(img.__class__.__name__, nib.Nifti1Image.__name__)
    npt.assert_equal(img.get_fdata().dtype, np.float64)
    npt.assert_equal(img.get_fdata().shape, (193, 229, 193))


def _check_tracking_evaluation_config(config):

    for key, val in config.items():
        assert set(tracking_config_file_necessary_keys) <= val.keys()


def test_check_hash():

    _, fname = tempfile.mkstemp()

    stored_hash = fetcher._get_file_hash(fname)

    # If all is well, this shouldn't return anything
    npt.assert_equal(fetcher.check_hash(fname, stored_hash), None)

    # If None is provided as input, it should silently not check either
    npt.assert_equal(fetcher.check_hash(fname, None), None)

    # Otherwise, it will raise its exception class
    npt.assert_raises(fetcher.FetcherError, fetcher.check_hash, fname, "foo")


def test_make_fetcher():

    # Make a fetcher with some test data using a local server
    with TemporaryDirectory() as tmpdir:

        test_data = TEST_FILES["fibercup_T1w"]
        name = "fetch_fibercup_test_data"
        remote_fnames = [op.sep + op.split(test_data)[-1]]
        local_fnames = ["fibercup_name"]
        doc = "Download Fiber Cup dataset anatomy data"
        data_size = "543B"
        msg = None
        unzip = False

        stored_hash = fetcher._get_file_hash(test_data)

        # Create local HTTP Server
        testfile_folder = op.split(test_data)[0] + os.sep
        testfile_url = "file:" + pathname2url(testfile_folder)
        current_dir = os.getcwd()
        # Change pwd to directory containing testfile
        os.chdir(testfile_folder)
        server = HTTPServer(("localhost", 8000), SimpleHTTPRequestHandler)
        server_thread = Thread(target=server.serve_forever)
        server_thread.deamon = True
        server_thread.start()

        # Test make_fetcher
        data_fetcher = fetcher._make_fetcher(
            name,
            tmpdir,
            testfile_url,
            remote_fnames,
            local_fnames,
            hash_list=[stored_hash],
            doc=doc,
            data_size=data_size,
            msg=msg,
            unzip=unzip,
        )

        try:
            data_fetcher()
        except Exception as e:
            print(e)
            # Stop local HTTP Server
            server.shutdown()

        assert op.isfile(op.join(tmpdir, local_fnames[0]))

        npt.assert_equal(
            fetcher._get_file_hash(op.join(tmpdir, local_fnames[0])),
            stored_hash,
        )

        # Stop local HTTP Server
        server.shutdown()
        # Change to original working directory
        os.chdir(current_dir)

    # Make a fetcher with actual data storage
    with TemporaryDirectory() as tmpdir:

        name = "fetch_fibercup_dwi"
        remote_fnames = ["download"]
        local_fnames = ["sub01-dwi.zip"]
        doc = "Download Fiber Cup dataset diffusion data"
        data_size = "0.39MB"
        msg = None
        unzip = True

        stored_hash = "705396981f1bcda51de12098db968390"

        rel_data_folder = pjoin("datasets", "fibercup", "raw", "sub-01", "dwi")

        folder = pjoin(tmpdir, rel_data_folder)
        testfile_url = TRACTODATA_DATASETS_URL + "br4ds/"

        data_fetcher = fetcher._make_fetcher(
            name,
            folder,
            testfile_url,
            remote_fnames,
            local_fnames,
            hash_list=[stored_hash],
            doc=doc,
            data_size=data_size,
            msg=msg,
            unzip=unzip,
        )

        try:
            files, folder = data_fetcher()
        except Exception as e:
            print(e)

        fnames = files["sub01-dwi.zip"][2]

        assert [op.isfile(op.join(tmpdir, f)) for f in fnames]

        npt.assert_equal(
            fetcher._get_file_hash(op.join(folder, local_fnames[0])),
            stored_hash,
        )


def test_fetch_data():

    # Fetch some test data using a local server
    with TemporaryDirectory() as tmpdir:

        test_data = TEST_FILES["fibercup_T1w"]

        stored_hash = fetcher._get_file_hash(test_data)
        bad_sha = "8" * len(stored_hash)

        newfile = op.join(tmpdir, "testfile.txt")
        # Test that the fetcher can get a file
        testfile_url = test_data
        testfile_folder, testfile_name = op.split(testfile_url)
        # Create local HTTP Server
        test_server_url = "http://127.0.0.1:8001/" + testfile_name
        current_dir = os.getcwd()
        # Change pwd to directory containing testfile
        os.chdir(testfile_folder + os.sep)
        server = HTTPServer(("localhost", 8001), SimpleHTTPRequestHandler)
        server_thread = Thread(target=server.serve_forever)
        server_thread.deamon = True
        server_thread.start()

        files = {"testfile.txt": (test_server_url, stored_hash)}
        try:
            fetcher.fetch_data(files, tmpdir)
        except Exception as e:
            print(e)
            # Stop local HTTP Server
            server.shutdown()

        npt.assert_(op.exists(newfile))

        # Test that the file is replaced when the hash doesn't match
        with open(newfile, "a") as f:
            f.write("some text")
        try:
            fetcher.fetch_data(files, tmpdir)
        except Exception as e:
            print(e)
            # stop local HTTP Server
            server.shutdown()

        npt.assert_(op.exists(newfile))
        npt.assert_equal(fetcher._get_file_hash(newfile), stored_hash)

        # Test that an error is raised when the hash of the downloaded file
        # does not match the expected value
        files = {"testfile.txt": (test_server_url, bad_sha)}
        npt.assert_raises(
            fetcher.FetcherError, fetcher.fetch_data, files, tmpdir
        )

        # Stop local HTTP Server
        server.shutdown()
        # Change to original working directory
        os.chdir(current_dir)


def test_get_fnames():

    for name in Dataset.__members__.keys():
        fetcher.get_fnames(name)


def test_tractodata_home():

    test_path = "TEST_PATH"
    if "TRACTODATA_HOME" in os.environ:
        old_home = os.environ["TRACTODATA_HOME"]
        del os.environ["TRACTODATA_HOME"]
    else:
        old_home = None

    reload(fetcher)

    npt.assert_string_equal(
        fetcher.tractodata_home,
        op.join(os.path.expanduser("~"), ".tractodata"),
    )
    os.environ["TRACTODATA_HOME"] = test_path
    reload(fetcher)
    npt.assert_string_equal(fetcher.tractodata_home, test_path)

    # Return to previous state
    if old_home:
        os.environ["TRACTODATA_HOME"] = old_home


def test_list_fibercup_bundles():

    bundle_names = fetcher.list_bundles_in_dataset(
        Dataset.FIBERCUP_SYNTH_BUNDLING.name
    )

    expected_val = len(fibercup_bundles)
    obtained_val = len(bundle_names)

    assert expected_val == obtained_val

    expected_val = fibercup_bundles
    obtained_val = bundle_names

    assert expected_val == obtained_val


def test_list_fibercup_bundle_masks():

    bundle_mask_names = fetcher.list_bundles_in_dataset(
        Dataset.FIBERCUP_BUNDLE_MASKS.name
    )

    expected_val = len(fibercup_bundles)
    obtained_val = len(bundle_mask_names)

    assert expected_val == obtained_val

    expected_val = fibercup_bundles
    obtained_val = bundle_mask_names

    npt.assert_equal(expected_val, obtained_val)


def test_list_fibercup_bundle_endpoint_masks():

    bundle_endpoint_masks = fetcher.list_bundle_endpoint_masks_in_dataset(
        Dataset.FIBERCUP_BUNDLE_ENDPOINT_MASKS.name
    )

    expected_val = len(fibercup_bundles) * 2
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    expected_val = _build_fibercup_bundle_endpoints()

    expected_val = [
        fetcher._build_bundle_endpoint_key(elem[0], elem[1])
        for elem in expected_val
    ]
    obtained_val = bundle_endpoint_masks

    npt.assert_equal(expected_val, obtained_val)


def test_list_fibercup_tissue_maps():

    tissue_names = fetcher.list_tissue_maps_in_dataset(
        Dataset.FIBERCUP_TISSUE_MAPS.name
    )

    expected_val = fibercup_tissues
    obtained_val = tissue_names
    assert expected_val == obtained_val


def test_list_hcp_tr_dti_maps():

    dti_map_names = fetcher.list_dti_maps_in_dataset(
        Dataset.HCP_TR_DTI_MAPS.name
    )

    expected_val = hcp_tr_dti_maps
    obtained_val = dti_map_names
    assert expected_val == obtained_val


def test_list_hcp_tr_exclude_include_maps_in_dataset():

    exclude_include_map_names = fetcher.list_exclude_include_maps_in_dataset(
        Dataset.HCP_TR_EXCLUDE_INCLUDE_MAPS.name
    )

    expected_val = hcp_tr_exclude_include_maps
    obtained_val = exclude_include_map_names
    assert expected_val == obtained_val


def test_list_hcp_tr_pve_maps_in_dataset():

    pve_map_names = fetcher.list_tissue_maps_in_dataset(
        Dataset.HCP_TR_PVE_MAPS.name
    )

    expected_val = hcp_tr_pve_maps
    obtained_val = pve_map_names
    assert expected_val == obtained_val


def test_list_ismrm2015_dti_maps():

    dti_map_names = fetcher.list_dti_maps_in_dataset(
        Dataset.ISMRM2015_DTI_MAPS.name
    )

    expected_val = ismrm2015_dti_maps
    obtained_val = dti_map_names
    assert expected_val == obtained_val


def test_list_ismrm2015_bundles():

    bundle_names = fetcher.list_bundles_in_dataset(
        Dataset.ISMRM2015_SYNTH_BUNDLING.name
    )

    expected_val = 25
    obtained_val = len(bundle_names)
    assert expected_val == obtained_val

    _expected_val = _build_ismrm2015_bundles()

    expected_val = []
    for elem in _expected_val:
        if isinstance(elem, str):
            _name = fetcher._build_bundle_key(elem)
        else:
            _name = fetcher._build_bundle_key(elem[0], hemisphere=elem[1])

        expected_val.append(_name)

    expected_val = sorted(expected_val)
    obtained_val = sorted(bundle_names)

    npt.assert_equal(expected_val, obtained_val)


def test_list_ismrm2015_bundle_masks():

    bundle_masks = fetcher.list_bundles_in_dataset(
        Dataset.ISMRM2015_BUNDLE_MASKS.name
    )

    expected_val = 25
    obtained_val = len(bundle_masks)
    assert expected_val == obtained_val

    _expected_val = _build_ismrm2015_bundles()

    expected_val = []
    for elem in _expected_val:
        if isinstance(elem, str):
            _name = fetcher._build_bundle_key(elem)
        else:
            _name = fetcher._build_bundle_key(elem[0], hemisphere=elem[1])

        expected_val.append(_name)

    expected_val = sorted(expected_val)
    obtained_val = sorted(bundle_masks)

    npt.assert_equal(expected_val, obtained_val)


def test_list_ismrm2015_bundle_endpoint_masks():

    bundle_endpoint_masks = fetcher.list_bundle_endpoint_masks_in_dataset(
        Dataset.ISMRM2015_BUNDLE_ENDPOINT_MASKS.name
    )

    expected_val = 25 * 2
    obtained_val = len(bundle_endpoint_masks)
    assert expected_val == obtained_val

    _expected_val = _build_ismrm2015_bundle_endpoints()

    expected_val = []
    for elem in _expected_val:
        if len(elem) == 2:
            _name = fetcher._build_bundle_endpoint_key(elem[0], elem[1])
        else:
            _name = fetcher._build_bundle_endpoint_key(
                elem[0], elem[2], hemisphere=elem[1]
            )

        expected_val.append(_name)

    expected_val = sorted(expected_val)
    obtained_val = sorted(bundle_endpoint_masks)

    npt.assert_equal(expected_val, obtained_val)


def test_list_ismrm2015_tissue_maps():

    tissue_names = fetcher.list_tissue_maps_in_dataset(
        Dataset.ISMRM2015_TISSUE_MAPS.name
    )

    expected_val = ismrm2015_tissues
    obtained_val = tissue_names
    assert expected_val == obtained_val


def test_list_ismrm2015_surfaces():

    surface_names = fetcher.list_surfaces_in_dataset(
        Dataset.ISMRM2015_SURFACES.name
    )

    expected_val = _build_ismrm2015_surfaces()
    expected_val = [
        fetcher._build_surface_key(elem[0], elem[1]) for elem in expected_val
    ]
    obtained_val = surface_names

    assert expected_val == obtained_val


def test_read_fibercup_anat():

    anat_img = fetcher.read_dataset_anat(Dataset.FIBERCUP_ANAT.name)

    _check_fibercup_img(anat_img)


def test_read_fibercup_dwi():

    dwi_img, gtab = fetcher.read_dataset_dwi(Dataset.FIBERCUP_DWI.name)

    npt.assert_equal(dwi_img.__class__.__name__, nib.Nifti1Image.__name__)
    npt.assert_equal(dwi_img.get_fdata().dtype, np.float64)
    npt.assert_equal(dwi_img.get_fdata().shape, (64, 64, 3, 31))

    expected_val = 31
    obtained_val = len(gtab.bvals)

    assert expected_val == obtained_val

    expected_val = (0, 1000)

    assert np.logical_and(
        gtab.bvals >= expected_val[0], gtab.bvals <= expected_val[-1]
    ).all()

    expected_val = 31
    obtained_val = len(gtab.bvecs)

    assert expected_val == obtained_val


def test_read_fibercup_tissue_maps():

    tissue_name = [Tissue.WM.value]
    wm_img = fetcher.read_dataset_tissue_maps(
        Dataset.FIBERCUP_TISSUE_MAPS.name
    )[tissue_name[0]]

    _check_fibercup_img(wm_img)

    wm_img = fetcher.read_dataset_tissue_maps(
        Dataset.FIBERCUP_TISSUE_MAPS.name, tissue_name=tissue_name
    )

    _check_fibercup_img(wm_img[tissue_name[0]])


def test_read_fibercup_synth_tracking():

    sft = fetcher.read_dataset_tracking(
        Dataset.FIBERCUP_ANAT.name, Dataset.FIBERCUP_SYNTH_TRACKING.name
    )

    npt.assert_equal(sft.__class__.__name__, StatefulTractogram.__name__)

    expected_val = 7833
    obtained_val = len(sft)

    assert expected_val == obtained_val


def test_read_fibercup_synth_bundling():

    anat_name = Dataset.FIBERCUP_ANAT.name
    bundling_name = Dataset.FIBERCUP_SYNTH_BUNDLING.name

    bundles = fetcher.read_dataset_bundling(anat_name, bundling_name)

    expected_val = len(fibercup_bundles)
    obtained_val = len(bundles)

    assert expected_val == obtained_val

    npt.assert_equal(
        bundles["bundle4"].__class__.__name__, StatefulTractogram.__name__
    )

    expected_val = 1413
    obtained_val = len(bundles["bundle4"])
    assert expected_val == obtained_val

    bundle_name = ["bundle5"]
    bundles = fetcher.read_dataset_bundling(
        anat_name, bundling_name, bundle_name=bundle_name
    )

    expected_val = 683
    obtained_val = len(bundles[bundle_name[0]])
    assert expected_val == obtained_val

    bundle_name = ["bundle4", "bundle5"]
    bundles = fetcher.read_dataset_bundling(
        anat_name, bundling_name, bundle_name=bundle_name
    )

    expected_val = 1413
    obtained_val = len(bundles[bundle_name[0]])
    assert expected_val == obtained_val

    expected_val = 683
    obtained_val = len(bundles[bundle_name[-1]])
    assert expected_val == obtained_val


def test_read_fibercup_synth_bundle_centroids():

    anat_name = Dataset.FIBERCUP_ANAT.name
    bundling_name = Dataset.FIBERCUP_SYNTH_BUNDLE_CENTROIDS.name

    centroids = fetcher.read_dataset_bundling(anat_name, bundling_name)

    expected_val = len(fibercup_bundles)
    obtained_val = len(centroids)

    assert expected_val == obtained_val

    npt.assert_equal(
        centroids["bundle6"].__class__.__name__, StatefulTractogram.__name__
    )

    expected_val = [1] * len(centroids)
    obtained_val = [len(centroid) for key, centroid in centroids.items()]

    assert expected_val == obtained_val


def test_read_fibercup_bundle_masks():

    name = Dataset.FIBERCUP_BUNDLE_MASKS.name

    bundle_masks = fetcher.read_dataset_bundle_masks(name)

    expected_val = len(fibercup_bundles)
    obtained_val = len(bundle_masks)

    assert expected_val == obtained_val

    mask_img = list(bundle_masks.values())[-1]

    _check_fibercup_img(mask_img)

    bundle_name = ["bundle1"]
    bundle_masks = fetcher.read_dataset_bundle_masks(
        name, bundle_name=bundle_name
    )

    mask_img = bundle_masks[bundle_name[0]]

    _check_fibercup_img(mask_img)

    bundle_name = ["bundle1", "bundle7"]
    bundle_masks = fetcher.read_dataset_bundle_masks(
        name, bundle_name=bundle_name
    )

    for name in bundle_name:
        mask_img = bundle_masks[name]

        _check_fibercup_img(mask_img)


def test_read_fibercup_bundle_endpoint_masks():

    name = Dataset.FIBERCUP_BUNDLE_ENDPOINT_MASKS.name

    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(name)

    expected_val = len(fibercup_bundles) * 2
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    mask_endpoint_img = list(bundle_endpoint_masks.values())[-1]

    _check_fibercup_img(mask_endpoint_img)

    bundle_name = ["bundle3"]
    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(
        name, bundle_name=bundle_name
    )

    expected_val = 2
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    _name = fetcher._build_bundle_endpoint_key(
        bundle_name[0], Endpoint.HEAD.value
    )
    mask_endpoint_img = bundle_endpoint_masks[_name]

    _check_fibercup_img(mask_endpoint_img)

    bundle_name = ["bundle3", "bundle4", "bundle6"]
    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(
        name, bundle_name=bundle_name
    )

    expected_val = len(bundle_name * 2)
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    for bname in bundle_name:
        for endpt in Endpoint:
            _name = fetcher._build_bundle_endpoint_key(bname, endpt.value)
            mask_endpoint_img = bundle_endpoint_masks[_name]

            _check_fibercup_img(mask_endpoint_img)

    bundle_name = ["bundle3"]
    endpoint_name = Endpoint.HEAD.value
    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(
        name, bundle_name=bundle_name, endpoint_name=endpoint_name
    )

    expected_val = 1
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    _name = fetcher._build_bundle_endpoint_key(bundle_name[0], endpoint_name)
    mask_endpoint_img = bundle_endpoint_masks[_name]

    _check_fibercup_img(mask_endpoint_img)

    bundle_name = ["bundle3", "bundle4", "bundle6"]
    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(
        name, bundle_name=bundle_name, endpoint_name=endpoint_name
    )

    expected_val = len(bundle_name)
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    for bname in bundle_name:
        _name = fetcher._build_bundle_endpoint_key(bname, endpoint_name)
        mask_endpoint_img = bundle_endpoint_masks[_name]

        _check_fibercup_img(mask_endpoint_img)


def test_read_fibercup_diffusion_peaks():

    peaks = fetcher.read_dataset_diffusion_peaks(
        Dataset.FIBERCUP_DIFFUSION_PEAKS.name
    )

    expected_val = (64, 64, 3, 15)
    obtained_val = peaks.shape

    assert expected_val == obtained_val


def test_read_fibercup_local_prob_tracking():

    sft = fetcher.read_dataset_tracking(
        Dataset.FIBERCUP_ANAT.name, Dataset.FIBERCUP_LOCAL_PROB_TRACKING.name
    )

    npt.assert_equal(sft.__class__.__name__, StatefulTractogram.__name__)

    expected_val = 5186
    obtained_val = len(sft)

    assert expected_val == obtained_val


def test_read_fibercup_local_prob_bundling():

    bundles = fetcher.read_dataset_bundling(
        Dataset.FIBERCUP_ANAT.name, Dataset.FIBERCUP_LOCAL_PROB_BUNDLING.name
    )

    expected_val = 1
    obtained_val = len(bundles)
    assert expected_val == obtained_val

    expected_val = 759
    obtained_val = len(bundles["bundle3"])
    assert expected_val == obtained_val

    npt.assert_equal(
        bundles["bundle3"].__class__.__name__, StatefulTractogram.__name__
    )


def test_read_fibercup_tracking_evaluation_config():

    tracking_evaluation_config = (
        fetcher.read_dataset_tracking_evaluation_config(
            Dataset.FIBERCUP_TRACKING_EVALUATION_CONFIG.name
        )
    )

    expected_val = len(fibercup_bundles)
    obtained_val = len(tracking_evaluation_config)

    assert expected_val == obtained_val

    _check_tracking_evaluation_config(tracking_evaluation_config)


def test_read_hcp_tr_anat():

    anat_img = fetcher.read_dataset_anat(Dataset.HCP_TR_ANAT.name)

    _check_hcp_tr_img(anat_img)


def test_read_hcp_tr_dti_maps():

    map_name = [DTIMap.FA.value]
    dti_map_img = fetcher.read_dataset_dti_maps(Dataset.HCP_TR_DTI_MAPS.name)

    _check_hcp_tr_img(dti_map_img[map_name[0]])

    dti_map_img = fetcher.read_dataset_dti_maps(
        Dataset.HCP_TR_DTI_MAPS.name, map_name
    )

    _check_hcp_tr_img(dti_map_img[map_name[0]])


def test_read_hcp_tr_exclude_include_maps():

    exclude_include_maps = fetcher.read_dataset_exclude_include_maps(
        Dataset.HCP_TR_EXCLUDE_INCLUDE_MAPS.name
    )

    _check_hcp_tr_img(exclude_include_maps[ExcludeIncludeMap.EXCLUDE.value])
    _check_hcp_tr_img(exclude_include_maps[ExcludeIncludeMap.INCLUDE.value])
    _check_hcp_tr_img(exclude_include_maps[ExcludeIncludeMap.INTERFACE.value])

    exclude_include_name = [ExcludeIncludeMap.EXCLUDE.value]
    exclude_include_maps = fetcher.read_dataset_exclude_include_maps(
        Dataset.HCP_TR_EXCLUDE_INCLUDE_MAPS.name,
        exclude_include_name=exclude_include_name,
    )

    _check_hcp_tr_img(exclude_include_maps[exclude_include_name[0]])


def test_read_hcp_tr_pve_maps():

    pve_maps = fetcher.read_dataset_tissue_maps(Dataset.HCP_TR_PVE_MAPS.name)

    _check_hcp_tr_img(pve_maps[Tissue.CSF.value])
    _check_hcp_tr_img(pve_maps[Tissue.GM.value])
    _check_hcp_tr_img(pve_maps[Tissue.WM.value])

    tissue_name = [Tissue.CSF.value]
    pve_maps = fetcher.read_dataset_tissue_maps(
        Dataset.HCP_TR_PVE_MAPS.name, tissue_name=tissue_name
    )

    _check_hcp_tr_img(pve_maps[tissue_name[0]])


def test_read_hcp_tr_surfaces():

    surfaces = fetcher.read_dataset_surfaces(Dataset.HCP_TR_SURFACES.name)

    expected_val = 4
    obtained_val = len(surfaces)
    assert expected_val == obtained_val

    as_polydata = False
    surface_type = ["pial"]
    hemisphere_name = "L"
    surface = fetcher.read_dataset_surfaces(
        Dataset.HCP_TR_SURFACES.name,
        surface_type=surface_type,
        hemisphere_name=hemisphere_name,
        as_polydata=as_polydata,
    )

    _name = fetcher._build_surface_key(
        surface_type[0], hemisphere=hemisphere_name
    )

    expected_val = 81920
    obtained_val = surface[_name].get_nb_triangles()
    assert expected_val == obtained_val

    expected_val = 40962
    obtained_val = surface[_name].get_nb_vertices()
    assert expected_val == obtained_val

    hemisphere_name = "R"
    surface = fetcher.read_dataset_surfaces(
        Dataset.HCP_TR_SURFACES.name,
        surface_type=surface_type,
        hemisphere_name=hemisphere_name,
        as_polydata=as_polydata,
    )

    _name = fetcher._build_surface_key(
        surface_type[0], hemisphere=hemisphere_name
    )

    expected_val = 81920
    obtained_val = surface[_name].get_nb_triangles()
    assert expected_val == obtained_val

    expected_val = 40962
    obtained_val = surface[_name].get_nb_vertices()
    assert expected_val == obtained_val

    surface_type = ["wm"]
    as_polydata = True
    hemisphere_name = "L"
    surface = fetcher.read_dataset_surfaces(
        Dataset.HCP_TR_SURFACES.name,
        surface_type=surface_type,
        hemisphere_name=hemisphere_name,
        as_polydata=as_polydata,
    )

    _name = fetcher._build_surface_key(
        surface_type[0], hemisphere=hemisphere_name
    )

    expected_val = (81920, 3)
    obtained_val = vtk_u.get_polydata_triangles(surface[_name]).shape
    assert expected_val == obtained_val

    expected_val = (40962, 3)
    obtained_val = vtk_u.get_polydata_vertices(surface[_name]).shape
    assert expected_val == obtained_val


def test_read_hcp_tr_pft_tracking():

    sft = fetcher.read_dataset_tracking(
        Dataset.HCP_TR_ANAT.name, Dataset.HCP_TR_PFT_TRACKING.name
    )

    npt.assert_equal(sft.__class__.__name__, StatefulTractogram.__name__)

    expected_val = 20000
    obtained_val = len(sft)

    assert expected_val == obtained_val


def test_read_ismrm2015_anat():

    anat_img = fetcher.read_dataset_anat(Dataset.ISMRM2015_ANAT.name)

    _check_ismrm2015_img(anat_img)


def test_read_ismrm2015_dwi():

    dwi_img, gtab = fetcher.read_dataset_dwi(Dataset.ISMRM2015_DWI.name)

    npt.assert_equal(dwi_img.__class__.__name__, nib.Nifti1Image.__name__)
    npt.assert_equal(dwi_img.get_fdata().dtype, np.float64)
    npt.assert_equal(dwi_img.get_fdata().shape, (90, 108, 90, 33))

    expected_val = 33
    obtained_val = len(gtab.bvals)

    assert expected_val == obtained_val

    expected_val = (0, 1000)

    assert np.logical_and(
        gtab.bvals >= expected_val[0], gtab.bvals <= expected_val[-1]
    ).all()

    expected_val = 33
    obtained_val = len(gtab.bvecs)

    assert expected_val == obtained_val


def test_read_ismrm2015_dwi_preproc():

    dwi_img, gtab = fetcher.read_dataset_dwi(
        Dataset.ISMRM2015_DWI_PREPROC.name
    )

    npt.assert_equal(dwi_img.__class__.__name__, nib.Nifti1Image.__name__)
    npt.assert_equal(dwi_img.get_fdata().dtype, np.float64)
    npt.assert_equal(dwi_img.get_fdata().shape, (180, 216, 180, 33))

    expected_val = 33
    obtained_val = len(gtab.bvals)

    assert expected_val == obtained_val

    expected_val = (0, 1000)

    assert np.logical_and(
        gtab.bvals >= expected_val[0], gtab.bvals <= expected_val[-1]
    ).all()

    expected_val = 33
    obtained_val = len(gtab.bvecs)

    assert expected_val == obtained_val


def test_read_ismrm2015_tissue_maps():

    tissue_name = [Tissue.WM.value]
    wm_img = fetcher.read_dataset_tissue_maps(
        Dataset.ISMRM2015_TISSUE_MAPS.name
    )[tissue_name[0]]

    _check_ismrm2015_img(wm_img)

    wm_img = fetcher.read_dataset_tissue_maps(
        Dataset.ISMRM2015_TISSUE_MAPS.name, tissue_name=tissue_name
    )

    _check_ismrm2015_img(wm_img[tissue_name[0]])


def test_read_ismrm2015_surfaces():

    surfaces = fetcher.read_dataset_surfaces(Dataset.ISMRM2015_SURFACES.name)

    expected_val = 2
    obtained_val = len(surfaces)
    assert expected_val == obtained_val

    as_polydata = False
    surface_type = ["pial"]
    hemisphere_name = "L"
    surface = fetcher.read_dataset_surfaces(
        Dataset.ISMRM2015_SURFACES.name,
        surface_type=surface_type,
        hemisphere_name=hemisphere_name,
        as_polydata=as_polydata,
    )

    _name = fetcher._build_surface_key(
        surface_type[0], hemisphere=hemisphere_name
    )

    expected_val = 138822
    obtained_val = surface[_name].get_nb_triangles()
    assert expected_val == obtained_val

    expected_val = 69413
    obtained_val = surface[_name].get_nb_vertices()
    assert expected_val == obtained_val

    hemisphere_name = "R"
    surface = fetcher.read_dataset_surfaces(
        Dataset.ISMRM2015_SURFACES.name,
        surface_type=surface_type,
        hemisphere_name=hemisphere_name,
        as_polydata=as_polydata,
    )

    _name = fetcher._build_surface_key(
        surface_type[0], hemisphere=hemisphere_name
    )

    expected_val = 140448
    obtained_val = surface[_name].get_nb_triangles()
    assert expected_val == obtained_val

    expected_val = 70226
    obtained_val = surface[_name].get_nb_vertices()
    assert expected_val == obtained_val

    as_polydata = True
    hemisphere_name = "L"
    surface = fetcher.read_dataset_surfaces(
        Dataset.ISMRM2015_SURFACES.name,
        surface_type=surface_type,
        hemisphere_name=hemisphere_name,
        as_polydata=as_polydata,
    )

    _name = fetcher._build_surface_key(
        surface_type[0], hemisphere=hemisphere_name
    )

    expected_val = (138822, 3)
    obtained_val = vtk_u.get_polydata_triangles(surface[_name]).shape
    assert expected_val == obtained_val

    expected_val = (69413, 3)
    obtained_val = vtk_u.get_polydata_vertices(surface[_name]).shape
    assert expected_val == obtained_val

    hemisphere_name = "R"
    surface = fetcher.read_dataset_surfaces(
        Dataset.ISMRM2015_SURFACES.name,
        surface_type=surface_type,
        hemisphere_name=hemisphere_name,
        as_polydata=as_polydata,
    )

    _name = fetcher._build_surface_key(
        surface_type[0], hemisphere=hemisphere_name
    )

    expected_val = (140448, 3)
    obtained_val = vtk_u.get_polydata_triangles(surface[_name]).shape
    assert expected_val == obtained_val

    expected_val = (70226, 3)
    obtained_val = vtk_u.get_polydata_vertices(surface[_name]).shape
    assert expected_val == obtained_val


def test_read_ismrm2015_dti_maps():

    map_name = [DTIMap.FA.value]
    dti_map_img = fetcher.read_dataset_dti_maps(
        Dataset.ISMRM2015_DTI_MAPS.name
    )

    _check_ismrm2015_img(dti_map_img[map_name[0]])

    dti_map_img = fetcher.read_dataset_dti_maps(
        Dataset.ISMRM2015_DTI_MAPS.name, map_name
    )

    _check_ismrm2015_img(dti_map_img[map_name[0]])


def test_read_ismrm2015_synth_tracking():

    sft = fetcher.read_dataset_tracking(
        Dataset.ISMRM2015_ANAT.name, Dataset.ISMRM2015_SYNTH_TRACKING.name
    )

    npt.assert_equal(sft.__class__.__name__, StatefulTractogram.__name__)

    expected_val = 200433
    obtained_val = len(sft)

    assert expected_val == obtained_val


def test_read_ismrm2015_synth_bundling():

    anat_name = Dataset.ISMRM2015_ANAT.name
    bundling_name = Dataset.ISMRM2015_SYNTH_BUNDLING.name

    bundles = fetcher.read_dataset_bundling(anat_name, bundling_name)

    expected_val = 25
    obtained_val = len(bundles)

    assert expected_val == obtained_val

    npt.assert_equal(
        bundles["CC"].__class__.__name__, StatefulTractogram.__name__
    )

    expected_val = 17993
    obtained_val = len(bundles["CC"])
    assert expected_val == obtained_val

    bundle_name = ["CST"]
    hemisphere_name = "L"
    bundles = fetcher.read_dataset_bundling(
        anat_name,
        bundling_name,
        bundle_name=bundle_name,
        hemisphere_name=hemisphere_name,
    )

    _name = fetcher._build_bundle_key(
        bundle_name[0], hemisphere=hemisphere_name
    )

    expected_val = 7217
    obtained_val = len(bundles[_name])
    assert expected_val == obtained_val

    bundle_name = ["CST"]
    bundles = fetcher.read_dataset_bundling(
        anat_name, bundling_name, bundle_name=bundle_name
    )

    _name = fetcher._build_bundle_key(
        bundle_name[0], hemisphere=Hemisphere.LEFT.value
    )

    expected_val = 7217
    obtained_val = len(bundles[_name])
    assert expected_val == obtained_val

    _name = fetcher._build_bundle_key(
        bundle_name[0], hemisphere=Hemisphere.RIGHT.value
    )

    expected_val = 10232
    obtained_val = len(bundles[_name])
    assert expected_val == obtained_val

    bundle_name = ["CC", "Fornix"]
    bundles = fetcher.read_dataset_bundling(
        anat_name, bundling_name, bundle_name=bundle_name
    )

    expected_val = 17993
    obtained_val = len(bundles[bundle_name[0]])
    assert expected_val == obtained_val

    expected_val = 3831
    obtained_val = len(bundles[bundle_name[-1]])
    assert expected_val == obtained_val

    bundle_name = ["CA", "Cing"]
    bundles = fetcher.read_dataset_bundling(
        anat_name, bundling_name, bundle_name=bundle_name
    )

    expected_val = 431
    obtained_val = len(bundles[bundle_name[0]])
    assert expected_val == obtained_val

    _name = fetcher._build_bundle_key(
        bundle_name[-1], hemisphere=Hemisphere.LEFT.value
    )

    expected_val = 14343
    obtained_val = len(bundles[_name])
    assert expected_val == obtained_val

    _name = fetcher._build_bundle_key(
        bundle_name[-1], hemisphere=Hemisphere.RIGHT.value
    )

    expected_val = 20807
    obtained_val = len(bundles[_name])
    assert expected_val == obtained_val

    bundle_name = ["MCP", "SLF"]
    hemisphere_name = "L"
    bundles = fetcher.read_dataset_bundling(
        anat_name,
        bundling_name,
        bundle_name=bundle_name,
        hemisphere_name=hemisphere_name,
    )

    assert bundle_name[0] not in bundles.keys()

    expected_val = 1
    obtained_val = len(bundles)
    assert expected_val == obtained_val

    _name = fetcher._build_bundle_key(
        bundle_name[-1], hemisphere=hemisphere_name
    )

    expected_val = 12497
    obtained_val = len(bundles[_name])
    assert expected_val == obtained_val


def test_read_ismrm2015_bundle_masks():

    name = Dataset.ISMRM2015_BUNDLE_MASKS.name

    bundle_masks = fetcher.read_dataset_bundle_masks(name)

    expected_val = 25
    obtained_val = len(bundle_masks)

    assert expected_val == obtained_val

    mask_img = list(bundle_masks.values())[-1]

    _check_ismrm2015_img(mask_img)

    bundle_name = ["CST"]
    bundle_masks = fetcher.read_dataset_bundle_masks(
        name, bundle_name=bundle_name
    )

    _name = fetcher._build_bundle_key(
        bundle_name[0], hemisphere=Hemisphere.LEFT.value
    )
    mask_img = bundle_masks[_name]

    _check_ismrm2015_img(mask_img)

    bundle_name = ["CST", "Fornix"]
    bundle_masks = fetcher.read_dataset_bundle_masks(
        name, bundle_name=bundle_name
    )

    expected_val = 3
    obtained_val = len(bundle_masks)
    assert expected_val == obtained_val

    mask_img = bundle_masks[bundle_name[-1]]

    _check_ismrm2015_img(mask_img)

    _name = fetcher._build_bundle_key(
        bundle_name[0], hemisphere=Hemisphere.RIGHT.value
    )
    mask_img = bundle_masks[_name]

    _check_ismrm2015_img(mask_img)

    bundle_name = ["CP", "CST"]
    hemisphere_name = Hemisphere.RIGHT.value
    bundle_masks = fetcher.read_dataset_bundle_masks(
        name, bundle_name=bundle_name, hemisphere_name=hemisphere_name
    )

    assert bundle_name[0] not in bundle_masks.keys()

    expected_val = 1
    obtained_val = len(bundle_masks)
    assert expected_val == obtained_val

    _name = fetcher._build_bundle_key(
        bundle_name[-1], hemisphere=hemisphere_name
    )
    mask_img = bundle_masks[_name]

    _check_ismrm2015_img(mask_img)


def test_read_ismrm2015_bundle_endpoint_masks():

    name = Dataset.ISMRM2015_BUNDLE_ENDPOINT_MASKS.name

    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(name)

    expected_val = 50
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    mask_endpoint_img = list(bundle_endpoint_masks.values())[-1]

    _check_ismrm2015_img(mask_endpoint_img)

    bundle_name = ["CST"]
    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(
        name, bundle_name=bundle_name
    )

    expected_val = 4
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    _name = fetcher._build_bundle_endpoint_key(
        bundle_name[0], Endpoint.HEAD.value, hemisphere=Hemisphere.LEFT.value
    )
    mask_endpoint_img = bundle_endpoint_masks[_name]

    _check_ismrm2015_img(mask_endpoint_img)

    bundle_name = ["OR"]
    endpoint_name = Endpoint.HEAD.value
    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(
        name, bundle_name=bundle_name, endpoint_name=endpoint_name
    )

    expected_val = 2
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    _name = fetcher._build_bundle_endpoint_key(
        bundle_name[0], endpoint_name, hemisphere=Hemisphere.LEFT.value
    )
    mask_endpoint_img = bundle_endpoint_masks[_name]

    _check_ismrm2015_img(mask_endpoint_img)

    bundle_name = ["CA", "CC", "Fornix", "MCP"]
    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(
        name, bundle_name=bundle_name, endpoint_name=endpoint_name
    )

    expected_val = len(bundle_name)
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    for bname in bundle_name:
        _name = fetcher._build_bundle_endpoint_key(bname, endpoint_name)
        mask_endpoint_img = bundle_endpoint_masks[_name]

        _check_ismrm2015_img(mask_endpoint_img)

    bundle_name = ["ICP", "SCP"]
    hemisphere_name = "R"
    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(
        name,
        bundle_name=bundle_name,
        hemisphere_name=hemisphere_name,
        endpoint_name=endpoint_name,
    )

    expected_val = 2
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    for bname in bundle_name:
        _name = fetcher._build_bundle_endpoint_key(
            bname, endpoint_name, hemisphere=hemisphere_name
        )
        mask_endpoint_img = bundle_endpoint_masks[_name]

        _check_ismrm2015_img(mask_endpoint_img)


def test_get_ismrm2015_submission_id_from_filenames():

    fnames = [
        "/path/to/ismrm2015_tractography_challenge_submission1-0_angular_error_results.csv",  # noqa E501
        "/path/to/ismrm2015_tractography_challenge_submission1-1_angular_error_results.csv",  # noqa E501
        "/path/to/ismrm2015_tractography_challenge_submission1-2_angular_error_results.csv",  # noqa E501
        "/path/to/ismrm2015_tractography_challenge_submission2-1_angular_error_results.csv",
    ]  # noqa E501
    expected_val = ["1-0", "1-1", "1-2", "2-1"]
    obtained_val = fetcher._get_ismrm2015_submission_id_from_filenames(fnames)

    assert expected_val == obtained_val


def test_classify_ismrm2015_submissions_results_files():

    (
        overall_scores_fname,
        angular_error_score_fnames,
        bundle_score_fnames,
    ) = fetcher._classify_ismrm2015_submissions_results_files()

    expected_val = pjoin(
        fetcher.tractodata_home,
        "datasets",
        "ismrm2015",
        "derivatives",
        "submission",
        "synth",
        "sub-02",
        "dwi",
        "ismrm2015_tractography_challenge_overall_results.csv",
    )
    obtained_val = overall_scores_fname

    assert expected_val == obtained_val

    expected_val = 96
    obtained_val = len(angular_error_score_fnames)

    assert expected_val == obtained_val

    expected_val = 96
    obtained_val = len(bundle_score_fnames)

    assert expected_val == obtained_val


def test_read_ismrm2015_submissions_overall_performance_data():

    df = fetcher.read_ismrm2015_submissions_overall_performance_data()

    expected_val = 96
    obtained_val = len(df.index)

    assert expected_val == obtained_val

    expected_val = 6
    obtained_val = len(df.columns)

    assert expected_val == obtained_val

    score = ["IB"]
    df = fetcher.read_ismrm2015_submissions_overall_performance_data(
        score=score
    )

    expected_val = 96
    obtained_val = len(df.index)

    assert expected_val == obtained_val

    expected_val = score
    obtained_val = df.columns.to_list()

    assert expected_val == obtained_val


def test_read_ismrm2015_submissions_angular_performance_data():

    df = fetcher.read_ismrm2015_submissions_angular_performance_data()

    expected_val = 96 * 3
    obtained_val = len(df.index)

    assert expected_val == obtained_val

    expected_val = 5
    obtained_val = len(df.columns)

    assert expected_val == obtained_val

    score = ["Median"]
    df = fetcher.read_ismrm2015_submissions_angular_performance_data(
        score=score
    )

    expected_val = 96 * 3
    obtained_val = len(df.index)

    assert expected_val == obtained_val

    expected_val = score
    obtained_val = df.columns.to_list()

    assert expected_val == obtained_val

    roi = ["Voxels with crossing fibers"]
    df = fetcher.read_ismrm2015_submissions_angular_performance_data(roi=roi)

    expected_val = 96
    obtained_val = len(df.index)

    assert expected_val == obtained_val

    expected_val = 5
    obtained_val = len(df.columns)

    assert expected_val == obtained_val

    score = ["Standard deviation"]
    roi = ["Voxels with single fiber population"]
    df = fetcher.read_ismrm2015_submissions_angular_performance_data(
        score=score, roi=roi
    )

    expected_val = 96
    obtained_val = len(df.index)

    assert expected_val == obtained_val

    expected_val = score
    obtained_val = df.columns.to_list()

    assert expected_val == obtained_val


def test_read_ismrm2015_submissions_bundle_performance_data():

    df = fetcher.read_ismrm2015_submissions_bundle_performance_data()

    expected_val = 96 * 25
    obtained_val = len(df.index)

    assert expected_val == obtained_val

    expected_val = 4
    obtained_val = len(df.columns)

    assert expected_val == obtained_val

    score = ["Overlap (% of GT)"]
    df = fetcher.read_ismrm2015_submissions_bundle_performance_data(
        score=score
    )

    expected_val = 96 * 25
    obtained_val = len(df.index)

    assert expected_val == obtained_val

    expected_val = score
    obtained_val = df.columns.to_list()

    assert expected_val == obtained_val

    bundle_name = ["Cingulum (right)"]
    df = fetcher.read_ismrm2015_submissions_bundle_performance_data(
        bundle_name=bundle_name
    )

    expected_val = 96
    obtained_val = len(df.index)

    assert expected_val == obtained_val

    expected_val = 4
    obtained_val = len(df.columns)

    assert expected_val == obtained_val

    score = ["Count"]
    bundle_name = ["Uncinate Fasciculus (right)"]
    df = fetcher.read_ismrm2015_submissions_bundle_performance_data(
        score=score, bundle_name=bundle_name
    )

    expected_val = 96
    obtained_val = len(df.index)

    assert expected_val == obtained_val

    expected_val = score
    obtained_val = df.columns.to_list()

    assert expected_val == obtained_val


def test_read_ismrm2015_tracking_evaluation_config():

    tracking_evaluation_config = (
        fetcher.read_dataset_tracking_evaluation_config(
            Dataset.ISMRM2015_TRACKING_EVALUATION_CONFIG.name
        )
    )

    expected_val = 25
    obtained_val = len(tracking_evaluation_config)

    assert expected_val == obtained_val

    _check_tracking_evaluation_config(tracking_evaluation_config)


def test_read_mni2009cnonlinsymm_anat():

    anat_img = fetcher.read_dataset_anat(Dataset.MNI2009CNONLINSYMM_ANAT.name)

    _check_mni2009cnonlinsymm_img(anat_img)


def test_read_mni2009cnonlinsymm_surfaces():

    surfaces = fetcher.read_dataset_surfaces(
        Dataset.MNI2009CNONLINSYMM_SURFACES.name
    )

    expected_val = 2
    obtained_val = len(surfaces)
    assert expected_val == obtained_val

    as_polydata = False
    surface_type = ["pial"]
    hemisphere_name = "L"
    surface = fetcher.read_dataset_surfaces(
        Dataset.MNI2009CNONLINSYMM_SURFACES.name,
        surface_type=surface_type,
        hemisphere_name=hemisphere_name,
        as_polydata=as_polydata,
    )

    _name = fetcher._build_surface_key(
        surface_type[0], hemisphere=hemisphere_name
    )

    expected_val = 308894
    obtained_val = surface[_name].get_nb_triangles()
    assert expected_val == obtained_val

    expected_val = 154449
    obtained_val = surface[_name].get_nb_vertices()
    assert expected_val == obtained_val

    hemisphere_name = "R"
    surface = fetcher.read_dataset_surfaces(
        Dataset.MNI2009CNONLINSYMM_SURFACES.name,
        surface_type=surface_type,
        hemisphere_name=hemisphere_name,
        as_polydata=as_polydata,
    )

    _name = fetcher._build_surface_key(
        surface_type[0], hemisphere=hemisphere_name
    )

    expected_val = 309130
    obtained_val = surface[_name].get_nb_triangles()
    assert expected_val == obtained_val

    expected_val = 154567
    obtained_val = surface[_name].get_nb_vertices()
    assert expected_val == obtained_val

    as_polydata = True
    hemisphere_name = "L"
    surface = fetcher.read_dataset_surfaces(
        Dataset.MNI2009CNONLINSYMM_SURFACES.name,
        surface_type=surface_type,
        hemisphere_name=hemisphere_name,
        as_polydata=as_polydata,
    )

    _name = fetcher._build_surface_key(
        surface_type[0], hemisphere=hemisphere_name
    )

    expected_val = (308894, 3)
    obtained_val = vtk_u.get_polydata_triangles(surface[_name]).shape
    assert expected_val == obtained_val

    expected_val = (154449, 3)
    obtained_val = vtk_u.get_polydata_vertices(surface[_name]).shape
    assert expected_val == obtained_val

    hemisphere_name = "R"
    surface = fetcher.read_dataset_surfaces(
        Dataset.MNI2009CNONLINSYMM_SURFACES.name,
        surface_type=surface_type,
        hemisphere_name=hemisphere_name,
        as_polydata=as_polydata,
    )

    _name = fetcher._build_surface_key(
        surface_type[0], hemisphere=hemisphere_name
    )

    expected_val = (309130, 3)
    obtained_val = vtk_u.get_polydata_triangles(surface[_name]).shape
    assert expected_val == obtained_val

    expected_val = (154567, 3)
    obtained_val = vtk_u.get_polydata_vertices(surface[_name]).shape
    assert expected_val == obtained_val
