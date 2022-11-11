# flake8: noqa
import os.path as osp

import ec.archs
import ec.data
import ec.models
import ec.metrics
import ec.losses
from basicsr.test import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
