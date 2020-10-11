#!/usr/bin/env python

"""distutils / setuptools helpers.
"""


class Bunch(object):
    def __init__(self, vars):
        for key, name in vars.items():
            if key.startswith("__"):
                continue
            self.__dict__[key] = name


def read_vars_from(info_file):
    """Read variables from Python text file.

    Parameters
    ----------
    info_file : str
        Filename of file to read.

    Returns
    -------
    info_vars : Bunch instance
        Bunch object where variables read from ``info_file`` appear as
        attributes.
    """

    # Use exec for compabibility with Python 3
    ns = {}
    with open(info_file, "rt") as fobj:
        exec(fobj.read(), ns)

    return Bunch(ns)
