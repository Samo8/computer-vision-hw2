#!/bin/bash

for file in $1/*
do
	printf 'converting image %s \n' $file
	convert $file -resize "1024x1024!" $file
done
