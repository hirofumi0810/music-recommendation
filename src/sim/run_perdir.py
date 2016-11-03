#! /usr/bin/env python
#-*- coding: utf-8 -*-

""" 類似度計算の実行 """

import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
import json

# my module
import emd

MODEL_PATH = '../../model/'  # path to 'model' directory
SIMILARITY_PATH = '../../similarity/'  # path to 'similarity' directory

##########################################
# モデルを読み込むときから並列のがいいかも
##########################################


def similarity(data):
    """ compute similarity """

    key = data[0]
    music_id_path = data[1]
    base_weight = data[2]
    base_mean = data[3]
    base_cov = data[4]

    weight_path = os.path.join(music_id_path, 'weight.csv')
    mean_path = os.path.join(music_id_path, 'mean.csv')
    cov_path = os.path.join(music_id_path, 'cov')
    if not os.path.isfile(weight_path) or not os.path.isfile(mean_path) or not os.path.isfile(cov_path):
        print('Error: There is no mdelo file!')
        return (key, 0)
    else:
        weight = np.loadtxt(weight_path, delimiter=',', skiprows=1)
        mean = np.loadtxt(mean_path, delimiter=',', skiprows=1)
        cov = np.array([np.loadtxt(os.path.join(cov_path, 'cov' + str(i + 1) + '.csv'),
                                   delimiter=',', skiprows=1)
                        for i in range(len(base_mean))])
        d = emd.emd(base_weight, base_mean, base_cov, weight, mean, cov)
        print(1 / d)

        return (key, 1 / d)


def main(dir_index, model, music_component, base_music_id):
    # ├ model
    # │  ├── 1000 (index)
    # │  ├── 1001
    # │  │   ├── GMM
    # │  │   ├── iGMM
    # │  │   └── LDA
    # │  │       └── music components
    # │  │               └ music_id
    # │  │                   └ parameter.csv
    # │  │ ：
    # │  │ ：
    # │  └── 1087
    # └ similarity
    #     ├── 1000 (index)
    #     ├── 1001
    #     │     └── *.json
    #     │ ：
    #     │ ：
    #     └── 1087

    # Rebuild to be working with one directory which is specified by parameter
    # in order to avoid OOM killer.

    # relative path
    base_music_id_path = os.path.join(MODEL_PATH, dir_index, model,
                                      music_component, base_music_id)
    similarity_index_path = os.path.join(SIMILARITY_PATH, dir_index)
    print('Base model file :', base_music_id_path)

    # read all 'index (model)' directories
    model_index_paths = [os.path.join(MODEL_PATH, model_index)
                         for model_index in os.listdir(MODEL_PATH)]
    model_index_paths.sort()
    if os.path.join(MODEL_PATH, '.DS_Store') in model_index_paths:
        model_index_paths.remove(os.path.join(MODEL_PATH, '.DS_Store'))

    # read all 'music id' directories
    music_id_paths = {}
    music_component_paths = [os.path.join(model_index_path, model, music_component)
                             for model_index_path in model_index_paths]
    for i in range(len(music_component_paths)):
        music_ids = os.listdir(music_component_paths[i])
        if '.DS_Store' in music_ids:
            music_ids.remove('.DS_Store')

        for j in range(len(music_ids)):
            music_id_paths[music_ids[j]] = os.path.join(music_component_paths[i],
                                                        music_ids[j])

    # read all parameters of base music
    base_mean = np.loadtxt(os.path.join(base_music_id_path, 'mean.csv'),
                           delimiter=',', skiprows=1)
    base_cov = np.array([np.loadtxt(os.path.join(base_music_id_path, 'cov', 'cov' + str(i + 1) + '.csv'),
                                    delimiter=',', skiprows=1)
                         for i in range(len(base_mean))])
    base_weight = np.loadtxt(os.path.join(base_music_id_path, 'weight.csv'),
                             delimiter=',', skiprows=1)

    # make 'index (similarity)' directory
    if not os.path.isdir(similarity_index_path):
        os.mkdir(similarity_index_path)

    # read json
    if not base_music_id + '.json' in os.listdir(similarity_index_path):
        sim_dict = {}
    else:
        json_path = os.path.join(similarity_index_path,
                                 base_music_id + '.json')
        file = open(json_path, 'r')
        sim_dict = json.load(file)
        file.close()

    # compute similarity (multi processing)
    data = []
    for music_id, music_id_path in music_id_paths.items():
        if not music_id in sim_dict.keys():
            print(music_id)
            data.append((music_id, music_id_path,
                         base_weight, base_mean, base_cov))

    core_num = multiprocessing.cpu_count()
    p = multiprocessing.Pool(core_num - 1)
    result = p.map(similarity, data)
    for i in range(len(result)):
        key = result[i][0]
        sim = result[i][1]
        sim_dict[key] = sim
    p.close()

    # save json
    file = open(os.path.join(similarity_index_path,
                             base_music_id + '.json'), 'w')
    json.dump(sim_dict, file)
    file.close()


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Specify base_dir_index, model, music_component, and base_music_id,.')
        print('Usage: python3 run_perdir.py dir_index model music_component base_music_id')
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
