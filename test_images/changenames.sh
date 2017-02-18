#!/bin/bash

FOLDER=$1

ls -v "$FOLDER" | cat -n | while read n f; do mv "$FOLDER/$f" "$FOLDER/frame_$n.jpeg" ; done
