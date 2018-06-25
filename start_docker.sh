#!/bin/bash
read -r -p "Load image from file? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
    docker load -i ${PWD}/docker_backups/medsecteam5.tar.gz
fi

# start container with current directory mounted as home:
read -r -p "Run image as container 'mst5'? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
    docker run -v ${PWD}:/home/medsecteam5 --name mst5 -p 5001:5001 -dit medsecteam5/medsecteam5:base
fi

# get a shell in the container:
# simple:
#   docker exec -u medsecteam5 -it mst5 /bin/bash
# as user in home:
read -r -p "Get interactive Shell? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
    docker exec -u medsecteam5 -it mst5 /bin/bash -c 'cd ~; exec "${SHELL:-sh}"'
fi

read -r -p "Commit changes on Container to image and backup? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
    # commit changes to base image
    docker commit --author medsecteam5 mst5 medsecteam5/medsecteam5:base
    # export docker edited docker image
    docker save medsecteam5/medsecteam5:base | gzip -c > ${PWD}/docker_backups/medsecteam5.tar.gz
fi

read -r -p "Stop container? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
    docker container stop mst5
fi

read -r -p "remove? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
    docker container rm mst5
fi
