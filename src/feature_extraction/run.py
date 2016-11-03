#! /usr/bin/env python
#-*- coding: utf-8 -*-

""" 特徴量抽出の実行 """

import os
import numpy as np
import librosa
import multiprocessing

# feature extraction
import extract_by_librosa as lib

# path
WAV_PATH = '../../wav/'  # path to 'wav' directory
HARMONIC_PATH = '../../harmonic/'  # path to 'harmonic' directory
PERCUSSIVE_PATH = '../../percussive/'  # path to 'percussive' directory
CSV_PATH = '../../csv/'  # path to 'csv' directory

# constant value
SAMPLING_RATE = 44100  # Hz

# features to extract
FEATURES = {'mfcc': False,  # ok
            'power': False,  # ok
            'chroma': False,  # ok
            'bpm': False,  # ok
            'tempogram': False,
            'mel spectrogram': False,  # ok
            'f0': False}


def save_to_csv(data, csv_index_path, feature, wavfile):
    """ save extracted features to csv files """

    feature_path = os.path.join(csv_index_path, feature)

    # make a directory if no directory
    if not os.path.isdir(feature_path):
        os.mkdir(feature_path)

    base, ext = os.path.splitext(wavfile)
    csvfile_path = os.path.join(feature_path, base + '.csv')

    # remove if already extracted
    if os.path.isfile(csvfile_path):
        os.remove(csvfile_path)

    np.savetxt(csvfile_path, data, delimiter=",", header=feature)
    print('Saved to ' + csvfile_path)


def extract_feature(data):
    """ extract features """

    wav_index_path = data[0]
    harmonic_index_path = data[1]
    percussive_index_path = data[2]
    csv_index_path = data[3]
    wavfile = data[4]
    music_id, ext = os.path.splitext(wavfile)

    # ignore if already extracted
    extract = {}
    for key, value in FEATURES.items():
        csvfile_path = os.path.join(csv_index_path, key, music_id + '.csv')
        # extract[key] = value and not os.path.isfile(csvfile_path)
        extract[key] = value

    # read wav (original) file
    if extract['mfcc'] or extract['power'] or extract['mel spectrogram']:

        wavfile_path = os.path.join(wav_index_path, wavfile)
        y = lib.read_wav_file(wavfile_path)

        # MFCC
        if extract['mfcc']:
            val_mfcc, delta1_mfcc, delta2_mfcc = lib.mfcc(y)
            save_to_csv(val_mfcc, csv_index_path, 'mfcc', wavfile)
            save_to_csv(delta1_mfcc, csv_index_path, 'delta1_mfcc', wavfile)
            save_to_csv(delta2_mfcc, csv_index_path, 'delta2_mfcc', wavfile)

        # power
        if extract['power']:
            val_power, delta1_power = lib.power(y)
            save_to_csv(val_power, csv_index_path, 'power', wavfile)
            save_to_csv(delta1_power, csv_index_path, 'delta1_power', wavfile)

        # mel spectrogram
        if extract['mel spectrogram']:
            log_S = lib.mel_spectrogram(y)
            save_to_csv(log_S, csv_index_path, 'mel spectrogram', wavfile)

    # read wav (harmonic) file
    if extract['f0'] or extract['chroma']:

        harmonic_file_path = os.path.join(harmonic_index_path, wavfile)

        if not os.path.isfile(harmonic_file_path):
            print('Error: There are no harmonic wav file! (Music id: ' + music_id + ')')
            return 0

        y_harmonic = lib.read_wav_file(harmonic_file_path)

        # F0
        if extract['f0']:
            val_f0, delta1_f0 = lib.f0(y_harmonic)
            save_to_csv(val_f0, csv_index_path, 'f0', wavfile)
            save_to_csv(delta1_f0, csv_index_path, 'delta1_f0', wavfile)

        # chroma
        if extract['chroma']:
            val_chroma = lib.chroma(y_harmonic)
            save_to_csv(val_chroma, csv_index_path, 'chroma', wavfile)

    # read wav (percussive) file
    if extract['bpm'] or extract['tempogram']:

        percussive_file_path = os.path.join(percussive_index_path, wavfile)

        if not os.path.isfile(percussive_file_path):
            print('Error: There are no percussive wav file! (Music id: ' + music_id + ')')
            return 0

        y_percussive = lib.read_wav_file(percussive_file_path)

        # BPM
        if extract['bpm']:
            val_bpm = lib.bpm(y_percussive)
            save_to_csv(np.array([[val_bpm]]), csv_index_path, 'bpm', wavfile)

        # tempogram
        if extract['tempogram']:
            val_tempogram = lib.tempogram(y_percussive)
            save_to_csv(val_tempogram, csv_index_path, 'tempogram', wavfile)


def main():
    # ├ wav, harmonic, percussive
    # │  ├── 1000 (index)
    # │  ├── 1001
    # │  │   └── *.wav
    # │  │ ：
    # │  │ ：
    # │  └── 1087
    # │
    # └ csv
    #     ├── 1000 (index)
    #     ├── 1001
    #     │   └── feature
    #     │           └ *.csv
    #     │ ：
    #     │ ：
    #     └── 1087

    # read all 'index (wav)' directories
    wav_index_dir = [dirname for dirname in os.listdir(WAV_PATH)]
    wav_index_dir.sort()
    harmonic_index_dir = [dirname for dirname in os.listdir(HARMONIC_PATH)]
    harmonic_index_dir.sort()
    percussive_index_dir = [dirname for dirname in os.listdir(PERCUSSIVE_PATH)]
    percussive_index_dir.sort()

    for i in range(len(wav_index_dir)):
        if wav_index_dir[i] != '.DS_Store':
            # relative path
            wav_index_path = os.path.join(WAV_PATH, wav_index_dir[i])
            harmonic_index_path = os.path.join(HARMONIC_PATH,
                                               harmonic_index_dir[i])
            percussive_index_path = os.path.join(PERCUSSIVE_PATH,
                                                 percussive_index_dir[i])
            csv_index_path = os.path.join(CSV_PATH, wav_index_dir[i])

            # read all wav files in the 'index (wav)' directory
            wavfiles = [filename for filename in os.listdir(wav_index_path)]
            wavfiles.sort()

            # make 'index (csv)' directory
            if not os.path.isdir(csv_index_path):
                os.mkdir(csv_index_path)

            if '.DS_Store' in wavfiles:
                wavfiles.remove('.DS_Store')

            # extract features (multi processing)
            core_num = multiprocessing.cpu_count()
            p = multiprocessing.Pool(core_num - 1)
            data = [(wav_index_path, harmonic_index_path, percussive_index_path, csv_index_path, wavfiles[j])
                    for j in range(len(wavfiles))]
            p.map(extract_feature, data)
            p.close()


def test():
    # filepath = '../../music/1000/100000013.m4a'
    filepath = '../../wav/1000/100000271.wav'
    y = lib.read_wav_file(filepath)
    y_harmonic, y_percussive = lib.harmonic_percussive(y)

    # duration = librosa.get_duration(y=y, sr=SAMPLING_RATE)
    # print('=====music duration=====')
    # print(str(duration) + '[sec]')

    # n_sample = len(y)
    # print('=====sample num=====')
    # print(str(n_sample) + '[sample]')

    # val_mfcc, delta1_mfcc, delta2_mfcc = lib.mfcc(y)
    # print('=====MFCC=====')
    # print(val_mfcc.shape)
    # print(delta1_mfcc.shape)
    # print(delta2_mfcc.shape)

    # val_power, delta1_power = lib.power(y)
    # print('=====power=====')
    # print(len(val_power[0]))
    # print(len(delta1_power[0]))

    # print('=====BPM=====')
    # print(lib.bpm(y_percussive))

    # print('=====mel spectrogram=====')
    # print(lib.mel_spectrogram(y).shape)

    val_f0, delta1_f0 = lib.f0(y_harmonic)
    print('=====F0=====')
    print(val_f0.shape)
    print(delta1_f0.shape)

    # print('=====chromagram=====')
    # print(lib.chroma(y_harmonic).shape)

    print('====tempogram=====')
    print(lib.tempogram(y).shape)

    # harmonic & percussive
    # librosa.output.write_wav('../../test_harmonic.wav', y_harmonic,
    #                          SAMPLING_RATE, norm=True)
    # librosa.output.write_wav('../../test_percussive.wav', y_percussive,
    #                          SAMPLING_RATE, norm=True)


if __name__ == '__main__':
    main()
    # test()
