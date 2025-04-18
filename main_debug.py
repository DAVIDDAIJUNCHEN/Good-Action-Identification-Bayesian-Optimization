#!/usr/bin/env  /mnt/users/daijun_chen/tools/miniconda3.10/install/envs/python3_huggingface/bin/python

import os
import sys
import argparse
import time
import numpy as np

from good_action.functions import *
from good_action.GPBO import GPBO
from good_action.utils import FUNC, ALGO


ALGOS = ['gpucb', 'pg', 'pi', 'eg', 'ei', 'ts', 'gs', 'mes', 'sts']
algo = "gpucb"
eps = 0.0
N_INITS = 3
NOISY = False

if __name__ == "__main__":
    func = eval(FUNC["ack6"])(noisy=NOISY)
    func_noiseless = eval(FUNC["ack6"])(noisy=False)
    func_bounds = func_noiseless.bounds
    X_init = np.random.uniform(func_bounds[:, 0], func_bounds[:, 1], size=(N_INITS, func_bounds.shape[0]))

    BO_test = GPBO(func_noiseless, func_bounds, algo, eps)
    BO_test.initiate(X_init)

    x_val_ori, y_obs = BO_test.sample_new_value()

    print(x_val_ori)

