#!/bin/bash

REGEX=$1
FOLDER=$2

convert -delay 10 -loop 0 -alpha set -dispose previous `ls -v $REGEX` $FOLDER/target.gif
