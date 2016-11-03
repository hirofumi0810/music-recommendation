#! /usr/bin/env python
#-*- coding: utf-8 -*-

""" m4aファイルをwavファイルに変換 """

import os
import sys
import librosa

# path
HARMONIC_PATH = '../../harmonic/'  # path to 'harmonic' directory
PERCUSSIVE_PATH = '../../percussive/'  # path to 'percussive' directory

# constant value
SAMPLING_RATE = 44100  # Hz


def harmonic_percussive(wav_index_path, harmonic_index_path, percussive_index_path, wavfile):
    """ separate into harmonic & percussive components """

    base, ext = os.path.splitext(wavfile)
    if ext == '.wav':
        input_path = os.path.join(wav_index_path, wavfile)
        output_path_harmonic = os.path.join(harmonic_index_path, wavfile)
        output_path_percussive = os.path.join(percussive_index_path, wavfile)

        print('Separating ' + input_path)

        # read a wav file
        y, sr = librosa.load(input_path, sr=SAMPLING_RATE)

        # separate to harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # save to wav files
        librosa.output.write_wav(output_path_harmonic, y_harmonic,
                                 SAMPLING_RATE, norm=True)
        librosa.output.write_wav(output_path_percussive, y_percussive,
                                 SAMPLING_RATE, norm=True)

    else:
        print('Error: Not wav file!')
        sys.exit(1)


def main_hpss(wavfile_path):
    # ├ wav
    # │  ├── 1000 (index)
    # │  ├── 1001
    # │  │   └── *.wav
    # │  │ ：
    # │  │ ：
    # │  └── 1087
    # │
    # └ harmonic, percussive
    #     ├── 1000 (index)
    #     ├── 1001
    #     │   └── *.wav
    #     │ ：
    #     │ ：
    #     └── 1087

    # relative path
    wav_index_path, wavfile = os.path.split(wavfile_path)
    dir_index = os.path.basename(wav_index_path)
    harmonic_index_path = os.path.join(HARMONIC_PATH, dir_index)
    percussive_index_path = os.path.join(PERCUSSIVE_PATH, dir_index)

    # separate to harmonic & percussive components
    harmonic_percussive(wav_index_path, harmonic_index_path,
                        percussive_index_path, wavfile)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Specify file.')
        sys.exit(1)

    main_hpss(sys.argv[1])
