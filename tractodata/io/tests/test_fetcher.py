#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
import os
import sys
import tempfile

import numpy.testing as npt

import tractodata.io.fetcher as fetcher

from nibabel.tmpdirs import TemporaryDirectory

from tractodata.io.fetcher import TRACTODATA_DATASETS_URL, Datasets


def test_check_sha():
    _, fname = tempfile.mkstemp()
    stored_sha = fetcher._get_file_sha(fname)
    # If all is well, this shouldn't return anything
    npt.assert_equal(fetcher.check_sha(fname, stored_sha), None)
    # If None is provided as input, it should silently not check either
    npt.assert_equal(fetcher.check_sha(fname, None), None)
    # Otherwise, it will raise its exception class
    npt.assert_raises(fetcher.FetcherError, fetcher.check_sha, fname, "foo")


def test_make_fetcher():

    with TemporaryDirectory() as tmpdir:

        name = "fetch_fibercup_anat"
        folder = pjoin(tmpdir, "datasets", "fibercup", "raw", "sub-01", "anat")
        testfile_url = TRACTODATA_DATASETS_URL + "/datasets" + "/fibercup" + "/raw" + "/sub-01" + "/anat",
        remote_fnames = "T1w.nii.gz"
        local_fnames = "T1w.nii.gz"
        sha_list=[file1_SHA]
        doc = "Download Fiber Cup dataset anatomy data",
        data_size="12KB"
        msg = None
        unzip = True

        data_fetcher = fetcher._make_fetcher(
            name, folder, testfile_url, remote_fnames, local_fnames,
            sha_list=[stored_md5], doc=doc, data_size=data_size, msg=msg,
            unzip=unzip)


def test_fetch_data():

    with TemporaryDirectory() as tmpdir:

        sha = "file1_SHA"
        files = {"T1w.nii.gz": (TRACTODATA_DATASETS_URL, sha)}
        folder = pjoin(tmpdir, "datasets", "fibercup", "raw", "sub-01", "anat")
        data_size = "12KB",

        fetcher.fetch_data(files, folder, data_size=None):


def test_get_fnames():

    for name in Dataset._member_names_:
        fetcher.get_fnames(name)


def test_dipy_home():

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

    # return to previous state
    if old_home:
        os.environ["TRACTODATA_HOME"] = old_home
