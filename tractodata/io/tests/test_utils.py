#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy.testing as npt

from tractodata.io.utils import \
    (Label, get_label_value_from_filename, filter_filenames_on_value,
     filter_list_on_list, is_subseq, is_subseq_of_any,
     get_longest_common_subseq)


def test_get_bundle_from_filename():

    fname = "/dir/subdir/sub01-T1w_hemi-L_space-orig_desc-synth_subset-Cing_tractography.nii.gz"  # noqa E501

    expected_val = "Cing"
    obtained_val = get_label_value_from_filename(fname, Label.BUNDLE)

    assert expected_val == obtained_val

    fname = "/dir/subdir/sub01-T1w_space-orig_desc-synth_tractography.nii.gz"

    expected_val = None
    obtained_val = get_label_value_from_filename(fname, Label.BUNDLE)

    assert expected_val == obtained_val


def test_get_endpoint_from_filename():

    fname = "/dir/sub01-T1w_hemi-L_space-orig_desc-synth_subset-Cing_part-head_tractography.trk"  # noqa E501

    expected_val = "head"
    obtained_val = get_label_value_from_filename(fname, Label.ENDPOINT)

    assert expected_val == obtained_val

    fname = "/dir/subdir/sub01-T1w_hemi-L_space-orig_desc-synth_subset-Cing_part-tail_tractography.trk"  # noqa E501

    expected_val = "tail"
    obtained_val = get_label_value_from_filename(fname, Label.ENDPOINT)

    assert expected_val == obtained_val

    fname = "/dir/sub01-dwi_space-orig_desc-synth_subset-CC_part-he_tractography.trk"  # noqa E501

    expected_val = None
    obtained_val = get_label_value_from_filename(fname, Label.ENDPOINT)

    assert expected_val == obtained_val

    fname = "/dir/sub01-dwi_space-orig_desc-synth_subset-CC_tractography.trk"  # noqa E501

    expected_val = None
    obtained_val = get_label_value_from_filename(fname, Label.ENDPOINT)

    assert expected_val == obtained_val


def test_get_hemisphere_from_filename():

    fname = "/dir/sub01-dwi_hemi-L_space-orig_desc-synth_subset-Cing_tractography.trk"  # noqa E501

    expected_val = "L"
    obtained_val = get_label_value_from_filename(fname, Label.HEMISPHERE)

    assert expected_val == obtained_val

    fname = "/dir/subdir/sub01-dwi_hemi-R_space-orig_desc-synth_subset-Cing_tractography.trk"  # noqa E501

    expected_val = "R"
    obtained_val = get_label_value_from_filename(fname, Label.HEMISPHERE)

    assert expected_val == obtained_val

    fname = "/dir/sub01-dwi_space-orig_desc-synth_subset-CC_tractography.trk"

    expected_val = None
    obtained_val = get_label_value_from_filename(fname, Label.HEMISPHERE)

    assert expected_val == obtained_val

    fname = "/dir/sub01-dwi_hemi-C_space-orig_desc-synth_subset-Cing_tractography.trk"  # noqa E501

    expected_val = None
    obtained_val = get_label_value_from_filename(fname, Label.HEMISPHERE)

    assert expected_val == obtained_val


def test_get_tissue_from_filename():

    fname = "/dir/sub01-T1w_space-orig_label-WM_dseg.nii.gz"

    expected_val = "WM"
    obtained_val = get_label_value_from_filename(fname, Label.TISSUE)

    assert expected_val == obtained_val

    fname = "/dir/sub01-T1w_space-orig_label-GM_dseg.nii.gz"

    expected_val = None
    obtained_val = get_label_value_from_filename(fname, Label.TISSUE)

    assert expected_val == obtained_val

    fname = "/dir/sub01-T1w_space-orig_label-WM_seg.nii.gz"

    expected_val = None
    obtained_val = get_label_value_from_filename(fname, Label.TISSUE)

    assert expected_val == obtained_val


def test_filter_list_on_list():

    primary_list = ["subset-CC", "subset-Cing", "subset-CST"]
    secondary_list = ["CC"]

    expected_val = ["subset-CC"]
    obtained_val = filter_list_on_list(primary_list, secondary_list)

    npt.assert_equal(expected_val, obtained_val)


def test_filter_filenames_on_value():

    fnames = [
        "sub01-dwi_space-orig_desc-synth_subset-CC_tractography.trk",
        "sub01-dwi_hemi-L_space-orig_desc-synth_subset-Cing_tractography.trk",
        "sub01-dwi_hemi-R_space-orig_desc-synth_subset-Cing_tractography.trk",
        "sub01-dwi_hemi-L_space-orig_desc-synth_subset-CST_tractography.trk",
        "sub01-dwi_hemi-R_space-orig_desc-synth_subset-CST_tractography.trk",
        "sub01-dwi_hemi-C_space-orig_desc-synth_subset-CST_tractography.trk"]
    label = Label.HEMISPHERE
    value = ["L"]

    expected_val = [
        "sub01-dwi_hemi-L_space-orig_desc-synth_subset-Cing_tractography.trk",
        "sub01-dwi_hemi-L_space-orig_desc-synth_subset-CST_tractography.trk"]
    obtained_val = filter_filenames_on_value(fnames, label, value)

    npt.assert_equal(expected_val, obtained_val)

    value = ["L", "R"]

    expected_val = [
        "sub01-dwi_hemi-L_space-orig_desc-synth_subset-Cing_tractography.trk",
        "sub01-dwi_hemi-L_space-orig_desc-synth_subset-CST_tractography.trk",
        "sub01-dwi_hemi-R_space-orig_desc-synth_subset-Cing_tractography.trk",
        "sub01-dwi_hemi-R_space-orig_desc-synth_subset-CST_tractography.trk"]
    obtained_val = filter_filenames_on_value(fnames, label, value)

    npt.assert_equal(expected_val, obtained_val)

    label = Label.BUNDLE
    value = ["Cing"]

    expected_val = [
        "sub01-dwi_hemi-L_space-orig_desc-synth_subset-Cing_tractography.trk",
        "sub01-dwi_hemi-R_space-orig_desc-synth_subset-Cing_tractography.trk"]
    obtained_val = filter_filenames_on_value(fnames, label, value)

    npt.assert_equal(expected_val, obtained_val)

    fnames = [
        "sub01-dwi_space-orig_desc-synth_subset-CC_tractography.trk",
        "sub01-dwi_hemi-L_space-orig_desc-synth_subset-Cing_tractography.trk",
        "sub01-dwi_hemi-R_space-orig_desc-synth_subset-Cing_tractography.trk",
        "sub01-dwi_hemi-L_space-orig_desc-synth_subset-CST_tractography.trk",
        "sub01-dwi_hemi-R_space-orig_desc-synth_subset-CST_tractography.trk"]
    value = ["Cing", "CST"]

    expected_val = [
        "sub01-dwi_hemi-L_space-orig_desc-synth_subset-Cing_tractography.trk",
        "sub01-dwi_hemi-R_space-orig_desc-synth_subset-Cing_tractography.trk",
        "sub01-dwi_hemi-L_space-orig_desc-synth_subset-CST_tractography.trk",
        "sub01-dwi_hemi-R_space-orig_desc-synth_subset-CST_tractography.trk"]
    obtained_val = filter_filenames_on_value(fnames, label, value)

    npt.assert_equal(expected_val, obtained_val)

    value = ["CC"]

    expected_val = [
        "sub01-dwi_space-orig_desc-synth_subset-CC_tractography.trk"]
    obtained_val = filter_filenames_on_value(fnames, label, value)

    npt.assert_equal(expected_val, obtained_val)


def test_is_subseq():

    possible_subseq = "ismrm2015_tractography_challenge"
    seq = \
        "ismrm2015_tractography_challenge_submission1-0_angular_error_results"
    expected_val = True
    obtained_val = is_subseq(possible_subseq, seq)

    assert expected_val == obtained_val

    possible_subseq = "ismrm2020"
    expected_val = False
    obtained_val = is_subseq(possible_subseq, seq)

    assert expected_val == obtained_val


def test_is_subseq_of_any():

    find = "ismrm2015_tractography"
    data = [
        "ismrm2015_tractography_challenge_submission1-0_angular_error_results",
        "ismrm2015_tractography_challenge_submission1-1_angular_error_results",
        "ismrm2015_tractography_challenge_submission1-2_angular_error_results",
        "ismrm2015_tractography_challenge_submission2-1_angular_error_results"]
    expected_val = True
    obtained_val = is_subseq_of_any(find, data)

    assert expected_val == obtained_val

    find = "ismrm2015_tractogram"
    expected_val = False
    obtained_val = is_subseq_of_any(find, data)

    assert expected_val == obtained_val


def test_get_longest_common_subseq():

    data = [
        "ismrm2015_tractography_challenge_submission1-0_individual_bundle_results",  # noqa E501
        "ismrm2015_tractography_challenge_submission1-1_individual_bundle_results",  # noqa E501
        "ismrm2015_tractography_challenge_submission1-2_individual_bundle_results",  # noqa E501
        "ismrm2015_tractography_challenge_submission2-1_individual_bundle_results"]  # noqa E501
    expected_val = "ismrm2015_tractography_challenge_submission"
    obtained_val = get_longest_common_subseq(data)

    assert expected_val == obtained_val

    data = ["1-0_angular_error_results", "1-1_angular_error_results",
            "1-2_angular_error_results", "1-3_angular_error_results"]
    expected_val = "_angular_error_results"
    obtained_val = get_longest_common_subseq(data)

    assert expected_val == obtained_val
