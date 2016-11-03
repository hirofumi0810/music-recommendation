#! /usr/bin/env python
#-*- coding: utf-8 -*-

""" m4aファイルをwavファイルに変換 """

import os
import sys
import subprocess
import librosa
import multiprocessing

# path
M4A_PATH = '../../music/'  # path to 'music'(m4a) directory
WAV_PATH = '../../wav/'  # path to 'wav' directory
HARMONIC_PATH = '../../harmonic/'  # path to 'harmonic' directory
PERCUSSIVE_PATH = '../../percussive/'  # path to 'percussive' directory

# constant value
SAMPLING_RATE = 44100  # Hz


def convert_to_wav(m4a_index_path, wav_index_path, m4afile):
    """ convert m4a to wav and save it """

    base, ext = os.path.splitext(m4afile)
    if ext == '.m4a':
        input_path = os.path.join(m4a_index_path, base + '.m4a')
        output_path = os.path.join(wav_index_path, base + '.wav')

        # ignore if already converted
        if not os.path.isfile(output_path):
            print('Converting ' + input_path + ' to ' + output_path)
            try:
                cmd = 'ffmpeg -i ' + input_path + ' ' + output_path
                ret = subprocess.call(cmd, shell=True)
            except subprocess.CalledProcessError(p):
                print('Error: cmd:%s returncode:%s' %
                      (p.cmd, p.returncode))
                sys.exit(1)
    else:
        print('Error: Not m4a file!')
        sys.exit(1)


def harmonic_percussive(data):
    """ separate into harmonic & percussive components """

    wav_index_path = data[0]
    harmonic_index_path = data[1]
    percussive_index_path = data[2]
    wavfile = data[3]

    base, ext = os.path.splitext(wavfile)
    if ext == '.wav':
        input_path = os.path.join(wav_index_path, wavfile)
        output_path_harmonic = os.path.join(harmonic_index_path, wavfile)
        output_path_percussive = os.path.join(percussive_index_path, wavfile)

        # ignore if already converted
        if not os.path.isfile(output_path_harmonic) or not os.path.isfile(output_path_harmonic):
            print('Separating ' + input_path)

            # read a wav file
            y, sr = librosa.load(input_path, sr=SAMPLING_RATE)

            # separate to harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)

            # save to wav files
            librosa.output.write_wav(output_path_harmonic,
                                     y_harmonic,
                                     SAMPLING_RATE,
                                     norm=True)
            librosa.output.write_wav(output_path_percussive,
                                     y_percussive,
                                     SAMPLING_RATE,
                                     norm=True)

    else:
        print('Error: Not wav file!')
        sys.exit(1)


def main_convert():
    # ├ music
    # │  ├── 1000 (index)
    # │  ├── 1001
    # │  │   └── *.m4a
    # │  │ ：
    # │  │ ：
    # │  └── 1087
    # │
    # └ wav, harmonic, percussive
    #     ├── 1000 (index)
    #     ├── 1001
    #     │   └── *.wav
    #     │ ：
    #     │ ：
    #     └── 1087

    # read all 'index (music)' directories
    m4a_index_dir = [dirname for dirname in os.listdir(M4A_PATH)]
    m4a_index_dir.sort()

    for i in range(len(m4a_index_dir)):
        if m4a_index_dir[i] != '.DS_Store':
            # relative path
            m4a_index_path = os.path.join(M4A_PATH, m4a_index_dir[i])
            wav_index_path = os.path.join(WAV_PATH, m4a_index_dir[i])

            # read all m4a files in the 'index (music)' directory
            m4afiles = [filename for filename in os.listdir(m4a_index_path)]
            m4afiles.sort()

            # make a directory
            if not os.path.isdir(wav_index_path):
                os.mkdir(wav_index_path)

            # convert m4a to wav
            for j in range(len(m4afiles)):
                if m4afiles[j] != '.DS_Store':
                    convert_to_wav(m4a_index_path, wav_index_path, m4afiles[j])


def main_hpss(dir_index):
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

    # Rebuild to be working with one directory which is specified by parameter
    # in order to avoid OOM killer.

    # relative path
    wav_index_path = os.path.join(WAV_PATH, dir_index)
    harmonic_index_path = os.path.join(HARMONIC_PATH, dir_index)
    percussive_index_path = os.path.join(PERCUSSIVE_PATH, dir_index)
    print('Wave file:', wav_index_path)

    # read all wav files in the 'index (wav)' directory
    wavfiles = [filename for filename in os.listdir(wav_index_path)]
    wavfiles.sort()

    # make a directory
    if not os.path.isdir(harmonic_index_path):
        os.mkdir(harmonic_index_path)
    if not os.path.isdir(percussive_index_path):
        os.mkdir(percussive_index_path)

    # separate to harmonic & percussive components (multi processing)
    if '.DS_Store' in wavfiles:
        wavfiles.remove('.DS_Store')

    core_num = multiprocessing.cpu_count()
    p = multiprocessing.Pool(core_num - 1)
    data = [(wav_index_path, harmonic_index_path,
             percussive_index_path, wavfiles[j]) for j in range(len(wavfiles))]
    p.map(harmonic_percussive, data)
    p.close()


if __name__ == '__main__':
    # main_convert()
    if len(sys.argv) < 2:
        print('Specify directory.')
        sys.exit(1)

    main_hpss(sys.argv[1])
