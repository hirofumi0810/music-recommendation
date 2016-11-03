#! /usr/bin/env python
#-*- coding: utf-8 -*-

""" 楽曲のメタデータのデータベース（sqlite）の操作 """

import sys
import sqlite3

DB_PATH = '../../meta/meta.db'
TABLE = 'tracks'


def read_database(key):
    """ read database """

    # connect database
    connector = sqlite3.connect(DB_PATH)
    cursor = connector.cursor()

    # extract attributes
    cursor.execute('PRAGMA TABLE_INFO(' + TABLE + ')')
    result = cursor.fetchall()
    attributes = result[0][1].split('\t')

    # read all data
    cursor.execute('select * from ' + TABLE)
    result = cursor.fetchall()

    # arrange data
    musicObj = {}
    if key in attributes:
        index = attributes.index(key)
        for i in range(len(result)):
            row = result[i][0].split('\t')
            attr = row[index]
            musicObj[attr] = row
    else:
        print('There is no such attribute.')
        sys.exit()

    # close
    cursor.close()
    connector.close()

    return (attributes, musicObj)


def search_music(attribute, query):
    """ search columns """

    # read database
    attributes, data = read_database(attribute)

    # print
    print('===== SEARCH RESULT =====')
    if query in data:
        for i in range(len(data[query])):
            print(attributes[i] + ': ' + data[query][i])
    else:
        print('There is no such music.')
        sys.exit()

    # print('===== DATA NUMBER =====')
    # print(len(result))


def main():
    global TABLE
    # TABLE = 'attibutes'
    search_music('ID', '100061772')

if __name__ == '__main__':
    main()
