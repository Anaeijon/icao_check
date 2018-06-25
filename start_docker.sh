#!/bin/bash

# start container with current directory mounted as home:
docker run -v ${PWD}:/home/medsecteam5 --name mst5 -p 5001:5001 -dit medsecteam5/medsecteam5:base

# get a shell in the container:
# simple: 
#   docker exec -u medsecteam5 -it mst5 /bin/bash
# as user in home:
docker exec -u medsecteam5 -it mst5 /bin/bash -c 'cd ~; exec "${SHELL:-sh}"'

