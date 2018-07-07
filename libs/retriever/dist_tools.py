# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Project: Region-DH
# Module: libs.retriever.dist_tools
# Copyright (c) 2018
# Written by: Franck FOTSO
# Licensed under MIT License
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np

def chi2_distance(vec_A, vec_B, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum(((np.array(vec_A, dtype='float32') - np.array(vec_B,  dtype='float32')) ** 2)
                     / (np.array(vec_A, dtype='float32') + np.array(vec_B, dtype='float32') + eps))

    # return the chi-squared distance
    return d