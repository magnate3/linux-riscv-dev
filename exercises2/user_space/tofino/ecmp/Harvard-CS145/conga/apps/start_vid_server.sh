#! /bin/bash

if [ ! -z $1 ]; then
	~/mininet/util/m $1 php -S $2:80 -t ~/resources/wwwroot
else
	echo "start_vid_server [host name] [host ip address]"
	echo "	host name: on which host do you want to run the video streaming server. Eg: h1."
	echo "	host ip address: the ip address of the video streaming server."
fi

