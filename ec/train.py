# flake8: noqa
import os.path as osp

import ec.archs
import ec.data
import ec.models
import ec.losses
import ec.metrics
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
