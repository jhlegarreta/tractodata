# -*- coding: utf-8 -*-

import enum
import os
import re


bundle_label = "subset"
discrete_segmentation_label = "dseg"
endpoint_label = "part"
hemisphere_label = "hemi"
general_label = "label"
label_value_separator = "-"


class Label(enum.Enum):
    BUNDLE = "bundle"
    ENDPOINT = "endpoint"
    HEMISPHERE = "hemisphere"
    TISSUE = "tissue"


class Endpoint(enum.Enum):
    HEAD = "head"
    TAIL = "tail"


class Hemisphere(enum.Enum):
    LEFT = "L"
    RIGHT = "R"


class Tissue(enum.Enum):
    WM = "WM"


class LabelError(Exception):
    pass


def _build_bundle_regex():
    """Build the bundle regex.

    Returns
    -------
    Bundle regex.
    """

    return '(?<=_' + bundle_label + label_value_separator + ')(.*?)(?=_)'


def _build_endpoint_regex():
    """Build the bundle mask endpoint regex.

    Returns
    -------
    Bundle mask endpoint regex.
    """

    return '(?<=_' + endpoint_label + label_value_separator + ')(' + \
           Endpoint.HEAD.value + '|' + Endpoint.TAIL.value + ')(?=_)'


def _build_hemisphere_regex():
    """Build the hemisphere regex.

    Returns
    -------
    Hemisphere regex.
    """

    return '(?<=_' + hemisphere_label + label_value_separator + ')[' + \
           Hemisphere.LEFT.value + Hemisphere.RIGHT.value + ']{1}(?=_)'


def _build_tissue_segmentation_regex():
    """Build the tissue segmentation regex.

    Returns
    -------
    Tissue segmentation regex.
    """

    return '(?<=_' + general_label + label_value_separator + ')' + \
           Tissue.WM.value + '(?=_' + discrete_segmentation_label + ')'


def _get_filename_root(fname):
    """Get the root name from the filename.

    Parameters
    ----------
    fname : string
        Filename.

    Returns
    -------
    Root name.
    """

    return os.path.basename(fname).split('.')[0]


def _unknown_label_msg(label):
    """Build a message indicating that label is not known.

    Returns
    -------
    msg : string
        Message.
    """

    msg = "Unknown label.\nProvided: {}; Available: {}".format(
        label, Label.__members__.values())
    return msg


def get_label_value_from_filename(fname, label):
    """Get the value to the relevant label contained in the filename.

    Parameters
    ----------
    fname : string
        Filename where to look for the label value.
    label : Label
        Label whose value needs to be sought.

    Returns
    -------
    label_value : None, string
        Label value. `None` if not found.
    """

    fname_root = _get_filename_root(fname)

    if label == Label.BUNDLE:
        regex = _build_bundle_regex()
    elif label == Label.ENDPOINT:
        regex = _build_endpoint_regex()
    elif label == Label.HEMISPHERE:
        regex = _build_hemisphere_regex()
    elif label == Label.TISSUE:
        regex = _build_tissue_segmentation_regex()
    else:
        raise LabelError(_unknown_label_msg(label))

    m = re.search(regex, fname_root)

    label_value = m

    if m:
        label_value = m.group(0)

    return label_value


def _build_label_value_pair(label, value):
    """Build a label value pair text.

    Parameters
    ----------
    label : str
        Label text.
    value : str
        Value text.

    Returns
    -------
    Label value text.
    """

    return label + label_value_separator + value


def filter_list_on_list(primary_list, secondary_list):
    """Filter the primary list based on the elements of the secondary list.

    Parameters
    ----------
    primary_list : list
        Strings to be filtered.
    secondary_list : list
        Strings to be sought in the primary list.

    Returns
    -------
     : list
        Elements in the primary list containing any element in the secondary
        list. An empty list is returned if no matches are found.
    """

    return list(filter(
        lambda item: any(x in item for x in secondary_list), primary_list))


def filter_filenames_on_value(fnames, label, value):
    """Filter filenames containing the given values for the given label.

    Parameters
    ----------
    fnames : list
        Filenames where to look for the label value.
    label : Label
        Label whose value needs to be sought.
    value : list
        List of label values whose value needs to be sought.

    Returns
    -------
    fname_shortlist : list
        Filenames containing the given label values. An empty list is returned
        if no matches are found.
    """

    fname_shortlist = []

    for _value in value:

        if label == Label.BUNDLE:
            label_value = _build_label_value_pair(bundle_label, _value)
        elif label == Label.ENDPOINT:
            label_value = _build_label_value_pair(endpoint_label, _value)
        elif label == Label.HEMISPHERE:
            label_value = _build_label_value_pair(hemisphere_label, _value)
        elif label == Label.TISSUE:
            label_value = _build_label_value_pair(
                discrete_segmentation_label, _value)
        else:
            raise LabelError(_unknown_label_msg(label))

        shortlist = filter_list_on_list(fnames, [label_value])
        fname_shortlist.extend(shortlist)

    return fname_shortlist


def is_subseq(possible_subseq, seq):
    """Determine whether the candidate string in a subsequence in the sequence.

    Parameters
    ----------
    possible_subseq : string
        Candidate string to be retrieved.
    seq : string
        Sequence where the candidate string is to be retrieved.

    Returns
    -------
    `True` if the candidate string is contained in the data; `False` otherwise.
    """

    def _get_length_n_slices(n):
        for i in range(len(seq) + 1 - n):
            yield seq[i:i+n]

    # Will also return True if possible_subseq == seq
    if len(possible_subseq) > len(seq):
        return False

    for slc in _get_length_n_slices(len(possible_subseq)):
        if slc == possible_subseq:
            return True

    return False


def is_subseq_of_any(find, data):
    """Determine whether the candidate string in a subsequence in all data
    elements.

    Parameters
    ----------
    find : string
        Candidate string to be retrieved.
    data : list
        Strings from which the longest common subsequence needs to be
        retrieved.

    Returns
    -------
    `True` if the candidate string is contained in all data elements; `False`
    otherwise.
    """

    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if not is_subseq(find, data[i]):
            return False

    return True


# ToDo
# These seem to have O(M*N) and apparently there are more efficient solutions
# using suffix trees (O(m+n)) or similar implementations (note that this may
# require # substrings other than a suffix) or difflib.
# Further references:
# https://stackoverflow.com/questions/40556491/how-to-find-the-longest-common-substring-of-multiple-strings
# https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
# or else, os.path.commonprefix
#
def get_longest_common_subseq(data):
    """Get longest common subsequence (starting from the beginning) across the
    given data.

    Parameters
    ----------
    data : list
        Strings from which the longest common subsequence needs to be
        retrieved.

    Returns
    -------
    substr : string
        The retrieved longest common subsequence.
    """

    substr = []

    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and is_subseq_of_any(data[0][i:i+j], data):
                    substr = data[0][i:i+j]

    return substr
