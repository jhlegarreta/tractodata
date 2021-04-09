"""Read test or example data."""

from os.path import join as pjoin, dirname


DATA_DIR = pjoin(dirname(__file__), 'datasets')

TEST_FILES = {
    'fibercup_T1w': pjoin(DATA_DIR, 'fibercup_T1w.nii.gz')
}
