#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import numpy.testing as npt

from tractodata.data import TEST_FILES


def test_test_data_dtypes():

    for data_name, data_path in TEST_FILES.items():
        img = nib.load(data_path)
        npt.assert_equal(img.get_fdata().dtype, np.float64)
