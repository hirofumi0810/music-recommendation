#! /usr/bin/env python
#-*- coding: utf-8 -*-

""" モデル学習の実行 """

import os
import sys
import numpy as np
import pandas as pd
import multiprocessing

# data processor
sys.path.append('../data_processor/')
import convert_to_matrix

# trainer
import train_model

CSV_PATH = '../../csv/'  # path to 'csv' directory
MODEL_PATH = '../../model/'  # path to 'model' directory

MODEL = 'igmm'
# {'gmm', 'igmm', 'lda'}

MUSIC_COMPONENT = 'timbre'
# {'timbre', 'harmony', 'tempo', 'rhythm', 'vocal'}

TIMBRE = ['mfcc', 'delta1_mfcc', 'delta2_mfcc',
          'power', 'delta1_power']
HARMONY = ['chroma']
TEMPO = []
RHYTHM = []
VOCAL = []

######################
# to do
# 音楽要素を変えれるように
######################


def save_model(parameters, model_index_path, model, music_component, music_id):
    """ save trained model (parameters) to csv files """

    model_path = os.path.join(model_index_path, model)

    # make a directory if no directory
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    music_component_path = os.path.join(model_path, music_component)

    # make a directory if no directory
    if not os.path.isdir(music_component_path):
        os.mkdir(music_component_path)

    music_id_path = os.path.join(music_component_path, music_id)

    # make a directory if no directory
    if not os.path.isdir(music_id_path):
        os.mkdir(music_id_path)

    # save to csv files
    for key, value in parameters.items():
        if key == 'cov':
            # make a directory if no directory
            if not os.path.isdir(os.path.join(music_id_path, key)):
                os.mkdir(os.path.join(music_id_path, key))

            for i in range(len(parameters[key])):
                np.savetxt(os.path.join(music_id_path, key, 'cov' + str(i + 1) + '.csv'),
                           parameters[key][i], delimiter=",", header='cov' + str(i + 1))
        else:
            np.savetxt(os.path.join(music_id_path, key + '.csv'),
                       parameters[key], delimiter=",",
                       header=model + ' / ' + music_component)

    print('Saved to ' + music_id_path)


def train(data):
    """ train model """

    X_train = data[0]
    model_index_path = data[1]
    csvfile = data[2]
    music_id, ext = os.path.splitext(csvfile)
    music_id_path = os.path.join(model_index_path, MODEL,
                                 MUSIC_COMPONENT, music_id)
    print('train model: ', music_id_path)

    # train model (ignore if already trained)
    if os.path.isdir(music_id_path):
        return 0
    elif MODEL == 'gmm':
        parameters = train_model.train_gmm(X_train, 30)
    elif MODEL == 'igmm':
        parameters = train_model.train_igmm(X_train, 50)
    elif MODEL == 'lda':
        parameters = train_model.train_lda(X_train)
    else:
        print('Error: There is no such model!')
        sys.exit(1)

    # save model
    save_model(parameters, model_index_path, MODEL, MUSIC_COMPONENT, music_id)


def main(dir_index):
    # ├ csv
    # │  ├── 1000 (index)
    # │  ├── 1001
    # │  │   └── feature
    # │  │           └ *.csv
    # │  │ ：
    # │  │ ：
    # │  └── 1087
    # │
    # └ model
    #     ├── 1000 (index)
    #     ├── 1001
    #     │   ├── GMM
    #     │   ├── iGMM
    #     │   └── LDA
    #     │       └── music components
    #     │               └ music_id
    #     │                   └ parameter.csv
    #     │ ：
    #     │ ：
    #     └── 1087

    # Rebuild to be working with one directory which is specified by parameter
    # in order to avoid OOM killer.

    # relative path
    csv_index_path = os.path.join(CSV_PATH, dir_index)
    model_index_path = os.path.join(MODEL_PATH, dir_index)
    print('CSV file :', csv_index_path)

    # read selected 'feature' directories
    if not os.path.isdir(csv_index_path):
        print('Error: There is no such index!')
        sys.exit(1)
    feature_dir_paths = [os.path.join(csv_index_path, TIMBRE[feat_i])
                         for feat_i in range(len(TIMBRE))]

    # read all csv files in the selected 'feature' directory
    # because all filenames are equal
    csvfiles = [filename for filename in os.listdir(feature_dir_paths[0])]
    csvfiles.sort()

    # make 'index (model)' directory
    if not os.path.isdir(model_index_path):
        os.mkdir(model_index_path)

    if '.DS_Store' in csvfiles:
        csvfiles.remove('.DS_Store')

    # data processing
    X_trains = []
    for i in range(len(csvfiles)):
        features = {}
        for j in range(len(feature_dir_paths)):
            csvfile_path = os.path.join(feature_dir_paths[j], csvfiles[i])
            feature = np.loadtxt(csvfile_path, delimiter=',', skiprows=1)
            features[TIMBRE[j]] = feature
        X_train = convert_to_matrix.timbre(features, TIMBRE)
        X_trains.append(X_train)

    # train model (multi processing)
    core_num = multiprocessing.cpu_count()
    p = multiprocessing.Pool(core_num - 1)
    data = [(X_trains[i], model_index_path, csvfiles[i])
            for i in range(len(csvfiles))]
    p.map(train, data)
    p.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Specify dir_index.')
        print('Usage: python3 run_perdir.py dir_index')
        sys.exit(1)

    main(sys.argv[1])
