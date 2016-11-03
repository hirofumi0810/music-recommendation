#! /usr/bin/env python
#-*- coding: utf-8 -*-

""" 類似度のランキング表示 """

import os
import sys
import json

SIMILARITY_PATH = '../../similarity/'  # path to 'similarity' directory


def main(dir_index, base_music_id):
    # └ similarity
    #     ├── 1000 (index)
    #     ├── 1001
    #     │     └── *.json
    #     │ ：
    #     │ ：
    #     └── 1087

    # relative path
    json_path = os.path.join(SIMILARITY_PATH, dir_index,
                             base_music_id + '.json')
    print('JSON file :', json_path)

    file = open(json_path, 'r')
    sim_dict = json.load(file)
    file.close()

    # sort by similarity
    for k, v in sorted(sim_dict.items(), key=lambda x: x[1]):
        print(k, v)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Specify dir_index and base_music_id,.')
        print('Usage: python3 run_perdir.py dir_index base_music_id')
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
