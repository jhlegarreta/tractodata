"""This file contains defines parameters for tractodata that we use to fill
settings in ``setup.py``, the tractodata top-level docstring.
"""

# tractodata version information. An empty _version_extra corresponds to a
# full release. ".dev" as a _version_extra string means this is a development
# version
_version_major = 0
_version_minor = 1
_version_micro = 0
_version_extra = "dev"
# _version_extra = ""

# Format expected by setup.py and doc/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

classifiers = ["Development Status :: 3 - Alpha",
               "Intended Audience :: Science/Research",
               "Programming Language :: Python :: 3",
               "Topic :: Scientific/Engineering",
               "Topic :: Scientific/Engineering ::Neuroimaging"]

description = "Tractography data repository to be used for " + \
              "tractography research."

keywords = "tractodata DWI DL ML neuroimaging tractography"

# Main setup parameters
NAME = "tractodata"
MAINTAINER = "jhlegarreta"
MAINTAINER_EMAIL = ""
DESCRIPTION = description
URL = "https://github.com/jhlegarreta/tractodata"
DOWNLOAD_URL = ""
BUG_TRACKER = "https://github.com/jhlegarreta/tractodata/issues",
DOCUMENTATION = "",
SOURCE_CODE = "https://github.com/jhlegarreta/tractodata",
LICENSE = ""
CLASSIFIERS = classifiers
KEYWORDS = keywords
AUTHOR = "jhlegarreta"
AUTHOR_EMAIL = ""
PLATFORMS = ""
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
ISRELEASE = _version_extra == ""
VERSION = __version__
PROVIDES = ["tractodata"]
REQUIRES = ["nibabel",
            "dipy"]
