#! /usr/bin/env python
#-*- coding: utf-8 -*-

""" モデルの学習 """

import sys

# model
sys.path.append('../model/')
import gmm
import igmm
import lda


def train_gmm(X_train, n_components):
    """ train GMM """

    # train parameters
    w, mean, cov = gmm.gmm(X_train, n_components)

    parameters = {}
    parameters['weight'] = w
    parameters['mean'] = mean
    covs = {}
    for i in range(len(w)):
        covs[i] = cov[i]
    parameters['cov'] = covs

    return parameters


def train_igmm(X_train, n_components):
    """ train infinite GMM """

    # train parameters
    w, mean, cov = igmm.igmm(X_train, n_components)

    parameters = {}
    parameters['weight'] = w
    parameters['mean'] = mean
    covs = {}
    for i in range(len(w)):
        covs[i] = cov[i]
    parameters['cov'] = covs

    return parameters


def train_lda(X_train):
    """ train LDA """

    # train parameters

    return 0
