#!/bin/sh

DIR_START=1050
DIR_END=1087
PYTHON=python3
# PYTHON=nohup python3

for DIR in `seq -w $DIR_START $DIR_END`; do
  # echo "Process starting for set $DIR..." >> ../../log/convert_1050.log
  # $PYTHON converter_perdir.py $DIR >> ../../log/convert_1050.log 2>&1
  echo "Process starting for set $DIR..."
  $PYTHON converter_perdir.py $DIR
done
