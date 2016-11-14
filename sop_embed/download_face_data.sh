#!/bin/bash
# Download the data
wget  pagesperso.litislab.fr/~sbelharbi/publishedCode/2016/pr-15-facial-landmarks-sop/inout/data/face.tar.gz -P ../inout/data/
# Untar the files
tar -zxvf ../inout/data/face.tar.gz  -C ../inout/data/
