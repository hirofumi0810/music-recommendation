#! /usr/bin/env python
#-*- coding: utf-8 -*-

""" wavファイルから特徴量を抽出 """

import numpy as np
import librosa

# constant value
SAMPLING_RATE = 44100  # Hz
WINDOW = 0.2  # sec
SHIFT = 0.1  # sec
N_FFT = int(SAMPLING_RATE * WINDOW)  # 8820
HOP_LENGTH = int(SAMPLING_RATE * SHIFT)  # 4410


def read_wav_file(filepath):
    """ read a wav file """

    y, sr = librosa.load(filepath, sr=SAMPLING_RATE)

    return y


##################################################
# for timbre
##################################################

def mfcc(y):
    """ compute MFCC """

    # compute log Mel spectrogram
    log_S = mel_spectrogram(y)

    # extract the top 13 Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(S=log_S, sr=SAMPLING_RATE, n_mfcc=13)
    # first delta
    delta1_mfcc = librosa.feature.delta(mfcc)
    # second delta
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    return (mfcc, delta1_mfcc, delta2_mfcc)


def power(y):
    """ compute RMS """

    # compute RMS
    rms = librosa.feature.rmse(y=y,
                               n_fft=N_FFT,
                               hop_length=HOP_LENGTH)

    # first delta
    delta1_rms = librosa.feature.delta(rms)

    return (rms, delta1_rms)


##################################################
# for harmony
##################################################

def harmonic_percussive(y):
    """ pull apart the harmonic and percussive components """

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    return (y_harmonic, y_percussive)


def chroma(y_harmonic):
    """ compute a CQT-based chromagram """

    # chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
    #                                         sr=SAMPLING_RATE,
    #                                         hop_length=HOP_LENGTH,
    #                                         n_chroma=12)

    chromagram = librosa.feature.chroma_stft(y=y_harmonic,
                                             sr=SAMPLING_RATE,
                                             n_fft=N_FFT,
                                             hop_length=HOP_LENGTH,
                                             n_chroma=12)

    return chromagram


##################################################
# for tempo
##################################################

def bpm(y_percussive):
    """ compute BPM """

    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=SAMPLING_RATE)

    return tempo


def tempogram(y):
    """ compute tempogram """

    # y_percussiveにするのか？？？？？
    tempogram = librosa.feature.tempogram(y,
                                          sr=SAMPLING_RATE,
                                          hop_length=HOP_LENGTH)

    return tempogram


##################################################
# for rhythm
##################################################

def mel_spectrogram(y):
    """ compute log Mel spectrogram """

    S = librosa.feature.melspectrogram(y,
                                       sr=SAMPLING_RATE,
                                       n_fft=N_FFT,
                                       hop_length=HOP_LENGTH,
                                       n_mels=40,
                                       fmax=8000)

    log_S = librosa.logamplitude(S, ref_power=np.max)

    return log_S


##################################################
# for vocal
##################################################

def f0(y_harmonic):
    """ compute F0 of melody """

    # compute F0
    f0, mag = librosa.piptrack(y=y_harmonic,
                               sr=SAMPLING_RATE,
                               n_fft=N_FFT,
                               hop_length=HOP_LENGTH,
                               fmin=150.0,
                               fmax=4000.0,
                               threshold=0.1)

    # first delta
    delta1_f0 = librosa.feature.delta(f0)

    return (f0, delta1_f0)
