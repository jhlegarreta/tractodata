#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as op
import os
import tempfile

import numpy.testing as npt

from http.server import HTTPServer, SimpleHTTPRequestHandler
from os.path import join as pjoin
from threading import Thread
from urllib.request import pathname2url

from importlib import reload

from nibabel.tmpdirs import TemporaryDirectory

from tractodata.data import TEST_FILES
import tractodata.io.fetcher as fetcher

from tractodata.io.fetcher import TRACTODATA_DATASETS_URL, Dataset


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
        local_fnames = ["dwi.zip"]
        doc = "Download Fiber Cup dataset anatomy data"
        data_size = "543B"
        msg = None
        unzip = True

        stored_hash = "f907901563254833c5f2bc90c209b4ae"

        rel_data_folder = pjoin("datasets", "fibercup", "raw", "sub-01",
                                "dwi")

        folder = pjoin(tmpdir, rel_data_folder)
        testfile_url = TRACTODATA_DATASETS_URL + "5yqvw/"

        data_fetcher = fetcher._make_fetcher(
            name, folder, testfile_url, remote_fnames, local_fnames,
            hash_list=[stored_hash], doc=doc, data_size=data_size, msg=msg,
            unzip=unzip)

        try:
            files, folder = data_fetcher()
        except Exception as e:
            print(e)

        fnames = files['dwi.zip'][2]

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

    for name in Dataset.__members__.values():
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
