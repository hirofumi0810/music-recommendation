#! /usr/bin/env python
#-*- coding: utf-8 -*-

""" EMD (Eatrh Mover's Distance) """

import numpy as np
import numpy.linalg
import pandas as pd
from pulp import *


def kl_div(mean1, cov1, mean2, cov2):
    """ compute KL divergence between Gaussian distributions (ignore constant term) """

    # compute inverse matrix
    try:
        inv_cov1 = np.linalg.inv(cov1)
    except numpy.linalg.linalg.LinAlgError:
        raise
    try:
        inv_cov2 = np.linalg.inv(cov2)
    except numpy.linalg.linalg.LinAlgError:
        raise

    # compute KL divergence
    term1 = np.sum(np.diag(np.dot(inv_cov2, cov1)))
    term2 = np.dot(np.dot((mean2 - mean1).T, inv_cov2), (mean2 - mean1))

    return term1 + term2


def sym_kl_div(mean1, cov1, mean2, cov2):
    """ compute symmetric KL divergence """

    kl_div1 = kl_div(mean1, cov1, mean2, cov2)
    kl_div2 = kl_div(mean2, cov2, mean1, cov1)
    val_sym_kl_div = 0.5 * (kl_div1 + kl_div2)

    return val_sym_kl_div


def emd(music1_w, music1_mean, music1_cov, music2_w, music2_mean, music2_cov):
    """ compute EMD """

    # number of components
    n_components_1 = len(music1_w)
    n_components_2 = len(music2_w)

    # compute distance matrix
    D = np.zeros(n_components_1 * n_components_2)
    for i in range(n_components_1):
        mean1 = music1_mean[i]
        cov1 = music1_cov[i]
        for j in range(n_components_2):
            mean2 = music2_mean[j]
            cov2 = music2_cov[j]
            # compute KL divergence between feature i and feature j
            D[i * n_components_2 + j] = sym_kl_div(mean1, cov1, mean2, cov2)

    # minimization problem
    m = LpProblem(sense=LpMinimize)
    f = [LpVariable('f%d' % i, lowBound=0, cat='Continuous')
         for i in range(n_components_1 * n_components_2)]

    # 目的関数
    m += lpDot(D, f)

    # 制約1
    for i in range(n_components_1):
        f_sum_2 = 0
        for j in range(n_components_2):
            f_sum_2 += f[i * n_components_2 + j]
        m += f_sum_2 <= music1_w[i]

    # 制約2
    for j in range(n_components_2):
        f_sum_1 = 0
        for i in range(n_components_1):
            f_sum_1 += f[i * n_components_2 + j]
        m += f_sum_1 <= music2_w[j]

    # 制約3
    f_sum = sum(f)
    m += f_sum == min(sum(music1_w), sum(music2_w))

    m.solve()

    return value(m.objective)
