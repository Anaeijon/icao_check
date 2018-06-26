#!/bin/bash
if [ ! -f ./neutralface/neutralface.run ]; then
	cd ./neutralface
  ./compile.sh
  cd ..
fi
python2 server.py
