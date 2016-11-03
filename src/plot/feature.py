#! /usr/bin/env python
#-*- coding: utf-8 -*-

""" 楽曲から抽出した特徴量のプロット """

import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

# my module
sys.path.append('../feature_extraction/')
import feature_extraction

# constant value
SAMPLING_RATE = 44100  # Hz
WINDOW = 0.2  # sec
SHIFT = 0.1  # sec
N_FFT = int(SAMPLING_RATE * WINDOW)  # 8820
HOP_LENGTH = int(SAMPLING_RATE * SHIFT)  # 4410


##################################################
# for timbre
##################################################

def plot_mfcc(y):
    """ plot MFCC """

    # compute MFCC, first delta, second delta
    mfcc, delta1_mfcc, delta2_mfcc = feature_extraction.mfcc(y)

    # plot
    plt.clf()
    plt.figure(figsize=(12, 6))
    # - MFCC
    plt.subplot(311)
    librosa.display.specshow(mfcc)
    plt.ylabel('MFCC')
    plt.colorbar()
    # - first delta
    plt.subplot(312)
    librosa.display.specshow(delta1_mfcc)
    plt.ylabel('MFCC-$\Delta$')
    plt.colorbar()
    # - second delta
    plt.subplot(313)
    librosa.display.specshow(delta2_mfcc, sr=SAMPLING_RATE, x_axis='time')
    plt.ylabel('MFCC-$\Delta^2$')
    plt.colorbar()

    plt.savefig('../../graph/' + 'mfcc.png', dvi=300)
    plt.tight_layout()


def plot_power(y):
    """ plot power """

    # compute log power spectrogram
    S, phase = librosa.magphase(librosa.stft(y,
                                             n_fft=N_FFT,
                                             hop_length=HOP_LENGTH))
    log_P = librosa.logamplitude(S**2, ref_power=np.max)

    # compute power, first delta
    power, delta1_power = feature_extraction.power(y)

    # plot
    plt.clf()
    plt.figure(figsize=(12, 6))
    # - power
    plt.subplot(311)
    plt.semilogy(power.T, label='RMS Energy')
    plt.xticks([])
    plt.xlim([0, power.shape[-1]])
    plt.legend(loc='best')
    # - first delta
    plt.subplot(312)
    plt.semilogy(delta1_power.T, label='RMS Energy-$\Delta$')
    plt.xticks([])
    plt.xlim([0, delta1_power.shape[-1]])
    plt.legend(loc='best')
    # - log power spectrogram
    plt.subplot(313)
    librosa.display.specshow(log_P, y_axis='log', x_axis='time')
    plt.title('log Power spectrogram')
    # plt.colorbar()

    plt.savefig('../../graph/' + 'power.png', dvi=300)
    plt.tight_layout()
    # なんで曲の途中までしか表示されない？
    # Hzが表示されてる


##################################################
# for harmony
##################################################

def plot_chroma(y):
    """ plot chromagram """

    # compute chromagram
    C = feature_extraction.chroma(y)

    # plot
    plt.clf()
    plt.figure(figsize=(12, 5))
    # Display the chromagram: the energy in each chromatic pitch class as a function of time
    # To make sure that the colors span the full range of chroma values,
    # set vmin and vmax
    librosa.display.specshow(C,
                             sr=SAMPLING_RATE,
                             x_axis='time',
                             y_axis='chroma',
                             vmin=0,
                             vmax=1)
    plt.title('Chromagram')
    plt.colorbar()

    plt.savefig('../../graph/' + 'chroma.png', dvi=300)
    plt.tight_layout()


##################################################
# for tempo
##################################################

def plot_tempogram(y):
    """ plot tempogram """

    # compute tempogram
    tempogram = feature_extraction.tempogram(y)

    # plot
    plt.clf()


##################################################
# for rhythm
##################################################

def plot_mel_spectrogram(y):
    """ plot Mel spectrogram """

    # compute Mel spectrogram
    log_S = feature_extraction.mel_spectrogram(y)

    # plot
    plt.clf()
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(log_S, sr=SAMPLING_RATE,
                             x_axis='time', y_axis='mel', fmax=8000)
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')

    plt.savefig('../../graph/' + 'mel_spectrogram.png', dvi=300)
    plt.tight_layout()


def plot_mel_spectrogram_harmonic_percussive(y):
    """ plot harmonic and percussive Mel spectrogram """

    # compute harmonic and percussive Mel spectrogram
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    log_Sh = feature_extraction.mel_spectrogram(y_harmonic)
    log_Sp = feature_extraction.mel_spectrogram(y_percussive)

    # plot
    plt.clf()
    plt.figure(figsize=(12, 6))
    # - harmonic
    plt.subplot(211)
    librosa.display.specshow(log_Sh, sr=SAMPLING_RATE, y_axis='mel')
    plt.title('mel power spectrogram (Harmonic)')
    plt.colorbar(format='%+02.0f dB')
    # - percussive
    plt.subplot(212)
    librosa.display.specshow(log_Sp, sr=SAMPLING_RATE,
                             x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram (Percussive)')
    plt.colorbar(format='%+02.0f dB')

    plt.savefig('../../graph/' + 'harmonic_percussive.png', dvi=300)
    plt.tight_layout()


##################################################
# for vocal
##################################################

def plot_f0(y):
    """ plot F0 """

    # compute F0, first delta
    f0, delta1_f0 = feature_extraction.f0(y)

    #️⃣ うまく表示されてない

    # plot
    plt.clf()
    plt.figure(figsize=(12, 6))
    # - F0
    plt.subplot(211)
    plt.semilogy(f0.T, label='RMS Energy')
    plt.xticks([])
    plt.xlim([0, f0.shape[-1]])
    plt.legend(loc='best')
    # - first delta
    # plt.subplot(211)
    # plt.semilogy(power.T, label='RMS Energy')
    # plt.xticks([])
    # plt.xlim([0, power.shape[-1]])
    # plt.legend(loc='best')

    # plt.title('log Power spectrogram')
    plt.savefig('../../graph/' + 'f0.png', dvi=300)
    plt.tight_layout()


def main():
    filepath = '../../wav/1000/100000271.wav'
    y = feature_extraction.read_wav_file(filepath)
    # plot_mfcc(y)
    plot_power(y)
    # plot_mel_spectrogram(y)
    # plot_mel_spectrogram_harmonic_percussive(y)

    plot_f0(y)
    # plot_chroma(y)
    # plot_tempogram(y)


if __name__ == '__main__':
    main()
