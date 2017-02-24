#!/bin/bash

FOLDER=$1

ls -v "$FOLDER" | cat -n | while read n f; do mv "$FOLDER/$f" `printf "$FOLDER/frame_%08d.jpeg" $n` ; done
