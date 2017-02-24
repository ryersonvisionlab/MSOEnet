#!/bin/bash

GIF1=$1
GIF2=$2
SIZE=$3
OUT=$4

convert \( $GIF1 -coalesce -append \) \( $GIF2 -coalesce -append \) -append -crop "$SIZE" +repage $OUT
