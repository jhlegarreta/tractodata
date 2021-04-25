#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import os.path as op
import os
import tempfile

import nibabel as nib
import numpy as np
import numpy.testing as npt

from http.server import HTTPServer, SimpleHTTPRequestHandler
from os.path import join as pjoin
from threading import Thread
from urllib.request import pathname2url

from importlib import reload

from dipy.io.streamline import StatefulTractogram
from nibabel.tmpdirs import TemporaryDirectory

from tractodata.data import TEST_FILES
import tractodata.io.fetcher as fetcher

from tractodata.io.fetcher import TRACTODATA_DATASETS_URL, Dataset
from tractodata.io.utils import Endpoint, Hemisphere, Tissue

fibercup_bundles = ["bundle1", "bundle2", "bundle3", "bundle4", "bundle5",
                    "bundle6", "bundle7"]
fibercup_tissues = [Tissue.WM.value]
ismrm2015_association_bundles = ["Cing", "FPT", "ICP", "ILF", "OR", "POPT",
                                 "SCP", "SLF", "UF"]
ismrm2015_projection_bundles = ["CST"]
ismrm2015_commissural_bundles = ["CC", "CA", "CP", "Fornix", "MCP"]
ismrm2015_tissues = [Tissue.WM.value]


def _build_fibercup_bundle_endpoints():

    return list(
        itertools.product(
            fibercup_bundles, [Endpoint.HEAD.value, Endpoint.TAIL.value]))


def _build_ismrm2015_bundles():

    assoc_val = list(itertools.product(
        ismrm2015_association_bundles,
        [Hemisphere.LEFT.value, Hemisphere.RIGHT.value]))

    proj_val = itertools.product(
        ismrm2015_projection_bundles,
        [Hemisphere.LEFT.value, Hemisphere.RIGHT.value])

    return list(itertools.chain.from_iterable(
        [assoc_val, ismrm2015_commissural_bundles, proj_val]))


def _check_fibercup_img(img):

    npt.assert_equal(
        img.__class__.__name__, nib.Nifti1Image.__name__)
    npt.assert_equal(img.get_fdata().dtype, np.float64)
    npt.assert_equal(img.get_fdata().shape, (64, 64, 3))


def _check_ismrm2015_img(img):

    npt.assert_equal(
        img.__class__.__name__, nib.Nifti1Image.__name__)
    npt.assert_equal(img.get_fdata().dtype, np.float64)
    npt.assert_equal(img.get_fdata().shape, (180, 216, 180))


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

        test_data = TEST_FILES['fibercup_T1w']
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
        testfile_url = 'file:' + pathname2url(testfile_folder)
        current_dir = os.getcwd()
        # Change pwd to directory containing testfile
        os.chdir(testfile_folder)
        server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
        server_thread = Thread(target=server.serve_forever)
        server_thread.deamon = True
        server_thread.start()

        # Test make_fetcher
        data_fetcher = fetcher._make_fetcher(
            name, tmpdir, testfile_url, remote_fnames, local_fnames,
            hash_list=[stored_hash], doc=doc, data_size=data_size, msg=msg,
            unzip=unzip)

        try:
            data_fetcher()
        except Exception as e:
            print(e)
            # Stop local HTTP Server
            server.shutdown()

        assert op.isfile(op.join(tmpdir, local_fnames[0]))

        npt.assert_equal(
            fetcher._get_file_hash(op.join(tmpdir, local_fnames[0])),
            stored_hash)

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

        rel_data_folder = pjoin("datasets", "fibercup", "raw", "sub-01",
                                "dwi")

        folder = pjoin(tmpdir, rel_data_folder)
        testfile_url = TRACTODATA_DATASETS_URL + "br4ds/"

        data_fetcher = fetcher._make_fetcher(
            name, folder, testfile_url, remote_fnames, local_fnames,
            hash_list=[stored_hash], doc=doc, data_size=data_size, msg=msg,
            unzip=unzip)

        try:
            files, folder = data_fetcher()
        except Exception as e:
            print(e)

        fnames = files['sub01-dwi.zip'][2]

        assert [op.isfile(op.join(tmpdir, f)) for f in fnames]

        npt.assert_equal(
            fetcher._get_file_hash(op.join(folder, local_fnames[0])),
            stored_hash)


def test_fetch_data():

    # Fetch some test data using a local server
    with TemporaryDirectory() as tmpdir:

        test_data = TEST_FILES['fibercup_T1w']

        stored_hash = fetcher._get_file_hash(test_data)
        bad_sha = '8' * len(stored_hash)

        newfile = op.join(tmpdir, "testfile.txt")
        # Test that the fetcher can get a file
        testfile_url = test_data
        testfile_folder, testfile_name = op.split(testfile_url)
        # Create local HTTP Server
        test_server_url = "http://127.0.0.1:8001/" + testfile_name
        current_dir = os.getcwd()
        # Change pwd to directory containing testfile
        os.chdir(testfile_folder + os.sep)
        server = HTTPServer(('localhost', 8001), SimpleHTTPRequestHandler)
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
        with open(newfile, 'a') as f:
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
        npt.assert_raises(fetcher.FetcherError,
                          fetcher.fetch_data, files, tmpdir)

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

    npt.assert_string_equal(fetcher.tractodata_home,
                            op.join(os.path.expanduser("~"), ".tractodata"))
    os.environ["TRACTODATA_HOME"] = test_path
    reload(fetcher)
    npt.assert_string_equal(fetcher.tractodata_home, test_path)

    # Return to previous state
    if old_home:
        os.environ["TRACTODATA_HOME"] = old_home


def test_list_fibercup_bundles():

    bundle_names = fetcher.list_bundles_in_dataset(
        Dataset.FIBERCUP_SYNTH_BUNDLING.name)

    expected_val = len(fibercup_bundles)
    obtained_val = len(bundle_names)

    assert expected_val == obtained_val

    expected_val = fibercup_bundles
    obtained_val = bundle_names

    assert expected_val == obtained_val


def test_list_fibercup_bundle_masks():

    bundle_mask_names = fetcher.list_bundles_in_dataset(
        Dataset.FIBERCUP_BUNDLE_MASKS.name)

    expected_val = len(fibercup_bundles)
    obtained_val = len(bundle_mask_names)

    assert expected_val == obtained_val

    expected_val = fibercup_bundles
    obtained_val = bundle_mask_names

    npt.assert_equal(expected_val, obtained_val)


def test_list_fibercup_bundle_endpoint_masks():

    bundle_endpoint_masks = fetcher.list_bundle_endpoint_masks_in_dataset(
        Dataset.FIBERCUP_BUNDLE_ENDPOINT_MASKS.name)

    expected_val = len(fibercup_bundles)*2
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    expected_val = _build_fibercup_bundle_endpoints()

    expected_val = [fetcher._build_bundle_endpoint_key(elem[0], elem[1])
                    for elem in expected_val]
    obtained_val = bundle_endpoint_masks

    npt.assert_equal(expected_val, obtained_val)


def test_list_fibercup_tissue_maps():

    tissue_names = fetcher.list_tissue_maps_in_dataset(
        Dataset.FIBERCUP_TISSUE_MAPS.name)

    expected_val = fibercup_tissues
    obtained_val = tissue_names
    assert expected_val == obtained_val


def test_list_ismrm2015_bundles():

    bundle_names = fetcher.list_bundles_in_dataset(
        Dataset.ISMRM2015_SYNTH_BUNDLING.name)

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
        gtab.bvals >= expected_val[0], gtab.bvals <= expected_val[-1]).all()

    expected_val = 31
    obtained_val = len(gtab.bvecs)

    assert expected_val == obtained_val


def test_read_fibercup_tissue_maps():

    wm_img = fetcher.read_dataset_tissue_maps(
        Dataset.FIBERCUP_TISSUE_MAPS.name)[Tissue.WM.value]

    _check_fibercup_img(wm_img)


def test_read_fibercup_synth_tracking():

    sft = fetcher.read_dataset_tracking(
        Dataset.FIBERCUP_ANAT.name, Dataset.FIBERCUP_SYNTH_TRACKING.name)

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
        bundles["bundle4"].__class__.__name__, StatefulTractogram.__name__)

    expected_val = 1413
    obtained_val = len(bundles["bundle4"])
    assert expected_val == obtained_val

    bundle_name = ["bundle5"]
    bundles = fetcher.read_dataset_bundling(
        anat_name, bundling_name, bundle_name=bundle_name)

    expected_val = 683
    obtained_val = len(bundles[bundle_name[0]])
    assert expected_val == obtained_val

    bundle_name = ["bundle4", "bundle5"]
    bundles = fetcher.read_dataset_bundling(
        anat_name, bundling_name, bundle_name=bundle_name)

    expected_val = 1413
    obtained_val = len(bundles[bundle_name[0]])
    assert expected_val == obtained_val

    expected_val = 683
    obtained_val = len(bundles[bundle_name[-1]])
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
        name, bundle_name=bundle_name)

    mask_img = bundle_masks[bundle_name[0]]

    _check_fibercup_img(mask_img)

    bundle_name = ["bundle1", "bundle7"]
    bundle_masks = fetcher.read_dataset_bundle_masks(
        name, bundle_name=bundle_name)

    for name in bundle_name:
        mask_img = bundle_masks[name]

        _check_fibercup_img(mask_img)


def test_read_fibercup_bundle_endpoint_masks():

    name = Dataset.FIBERCUP_BUNDLE_ENDPOINT_MASKS.name

    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(name)

    expected_val = len(fibercup_bundles)*2
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    mask_endpoint_img = list(bundle_endpoint_masks.values())[-1]

    _check_fibercup_img(mask_endpoint_img)

    bundle_name = ["bundle3"]
    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(
        name, bundle_name=bundle_name)

    expected_val = 2
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    _name = fetcher._build_bundle_endpoint_key(
        bundle_name[0], Endpoint.HEAD.value)
    mask_endpoint_img = bundle_endpoint_masks[_name]

    _check_fibercup_img(mask_endpoint_img)

    bundle_name = ["bundle3", "bundle4", "bundle6"]
    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(
        name, bundle_name=bundle_name)

    expected_val = len(bundle_name*2)
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
        name, bundle_name=bundle_name, endpoint_name=endpoint_name)

    expected_val = 1
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    _name = fetcher._build_bundle_endpoint_key(bundle_name[0], endpoint_name)
    mask_endpoint_img = bundle_endpoint_masks[_name]

    _check_fibercup_img(mask_endpoint_img)

    bundle_name = ["bundle3", "bundle4", "bundle6"]
    bundle_endpoint_masks = fetcher.read_dataset_bundle_endpoint_masks(
        name, bundle_name=bundle_name, endpoint_name=endpoint_name)

    expected_val = len(bundle_name)
    obtained_val = len(bundle_endpoint_masks)

    assert expected_val == obtained_val

    for bname in bundle_name:
        _name = fetcher._build_bundle_endpoint_key(bname, endpoint_name)
        mask_endpoint_img = bundle_endpoint_masks[_name]

        _check_fibercup_img(mask_endpoint_img)


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
        gtab.bvals >= expected_val[0], gtab.bvals <= expected_val[-1]).all()

    expected_val = 33
    obtained_val = len(gtab.bvecs)

    assert expected_val == obtained_val


def test_read_ismrm2015_synth_tracking():

    sft = fetcher.read_dataset_tracking(
        Dataset.ISMRM2015_ANAT.name, Dataset.ISMRM2015_SYNTH_TRACKING.name)

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
        bundles["CC"].__class__.__name__, StatefulTractogram.__name__)

    expected_val = 17993
    obtained_val = len(bundles["CC"])
    assert expected_val == obtained_val

    bundle_name = ["CST"]
    hemisphere_name = "L"
    bundles = fetcher.read_dataset_bundling(
        anat_name, bundling_name, bundle_name=bundle_name,
        hemisphere_name=hemisphere_name)

    _name = fetcher._build_bundle_key(
        bundle_name[0], hemisphere=hemisphere_name)

    expected_val = 7217
    obtained_val = len(bundles[_name])
    assert expected_val == obtained_val

    bundle_name = ["CST"]
    bundles = fetcher.read_dataset_bundling(
        anat_name, bundling_name, bundle_name=bundle_name)

    _name = fetcher._build_bundle_key(
        bundle_name[0], hemisphere=Hemisphere.LEFT.value)

    expected_val = 7217
    obtained_val = len(bundles[_name])
    assert expected_val == obtained_val

    _name = fetcher._build_bundle_key(
        bundle_name[0], hemisphere=Hemisphere.RIGHT.value)

    expected_val = 10232
    obtained_val = len(bundles[_name])
    assert expected_val == obtained_val

    bundle_name = ["CC", "Fornix"]
    bundles = fetcher.read_dataset_bundling(
        anat_name, bundling_name, bundle_name=bundle_name)

    expected_val = 17993
    obtained_val = len(bundles[bundle_name[0]])
    assert expected_val == obtained_val

    expected_val = 3831
    obtained_val = len(bundles[bundle_name[-1]])
    assert expected_val == obtained_val

    bundle_name = ["CA", "Cing"]
    bundles = fetcher.read_dataset_bundling(
        anat_name, bundling_name, bundle_name=bundle_name)

    expected_val = 431
    obtained_val = len(bundles[bundle_name[0]])
    assert expected_val == obtained_val

    _name = fetcher._build_bundle_key(
        bundle_name[-1], hemisphere=Hemisphere.LEFT.value)

    expected_val = 14343
    obtained_val = len(bundles[_name])
    assert expected_val == obtained_val

    _name = fetcher._build_bundle_key(
        bundle_name[-1], hemisphere=Hemisphere.RIGHT.value)

    expected_val = 20807
    obtained_val = len(bundles[_name])
    assert expected_val == obtained_val

    bundle_name = ["MCP", "SLF"]
    hemisphere_name = "L"
    bundles = fetcher.read_dataset_bundling(
        anat_name, bundling_name, bundle_name=bundle_name,
        hemisphere_name=hemisphere_name)

    assert bundle_name[0] not in bundles.keys()

    expected_val = 1
    obtained_val = len(bundles)
    assert expected_val == obtained_val

    _name = fetcher._build_bundle_key(
        bundle_name[-1], hemisphere=hemisphere_name)

    expected_val = 12497
    obtained_val = len(bundles[_name])
    assert expected_val == obtained_val
