#! /usr/bin/env python
#-*- coding: utf-8 -*-

""" 学習に使用する特徴量の選択・行列へ変換 """

import sys
import numpy as np


def timbre(features, timbre_contents):
    """ data processor for timbre

    Features
    ----------
    mfcc
    mfcc_delta1
    mfcc_delta2
    power
    power_delta1

    """

    if len(features) == len(timbre_contents):
        mfcc = features['mfcc']
        mfcc_delta1 = features['delta1_mfcc']
        mfcc_delta2 = features['delta2_mfcc']
        power = features['power'][:, np.newaxis].T
        power_delta1 = features['delta1_power'][:, np.newaxis].T

        return np.r_[mfcc, mfcc_delta1, mfcc_delta2, power, power_delta1].T

    else:
        print('Error: data length is not right.')
        sys.exit(1)


def harmony(data):
    """ data processor for harmony

    Features
    ----------
    chroma vectors

    """

    return 0


def tempo(data):
    """ data processor for tempo

    Features
    ----------

    """

    return 0


def rhythm(data):
    """ data processor for rhythm

    Features
    ----------

    """

    return 0


def vocal(data):
    """ data processor for vocal

    Features
    ----------

    """

    return 0
