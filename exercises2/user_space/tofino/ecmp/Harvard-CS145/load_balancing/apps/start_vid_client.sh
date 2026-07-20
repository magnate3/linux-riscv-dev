#!/bin/bash

if [ ! -z $1 ] || [ ! -z $2 ]; then
    chromium-browser --enable-logging --v=1 --user-data-dir=/home/p4/resources/chromedata/${1}-datadir http://$2/streaming.html --no-sandbox
else
    echo "start_vid_client [host name] [server ip address]"
    echo "  host name: on which host do you want to run the client. Eg: h1."
    echo "  server ip address: the ip address of the video streaming server."
fi
