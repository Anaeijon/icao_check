#!/bin/bash
if [ ! -f ./dlib/all/source.cpp ]; then
	wget http://dlib.net/files/dlib-19.9.tar.bz2
	tar --wildcards -xvjf dlib-*.tar.bz2 dlib-*/dlib
	mv dlib-*/dlib ./
	rm -r dlib-*/
fi
g++ -std=c++11 -O3 -I. ./dlib/all/source.cpp -lpthread -ljpeg -lpng -lX11 -DDLIB_JPEG_SUPPORT neutralface.cpp -o neutralface.run
