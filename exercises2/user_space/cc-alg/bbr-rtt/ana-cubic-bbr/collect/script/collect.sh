#!/usr/bin/env bash

set -o errexit; set -o pipefail; set -o nounset
trap "kill 0" EXIT

if [ "$#" -lt 1 ]; then
    echo "please pass some hosts to connect to"
    exit 1
fi

for SERVER in $@; do
    ssh -tt $SERVER "ss -H -n -tipa state connected exclude time-wait exclude fin-wait-2" |
    sed "s/127.0.0.1/$SERVER/g" > ss-$SERVER.txt &
    #replace local IP address with the external IP address we use to SSH in
    #this might work if you're SSHing in on the same interface as the connections you want to monitor
done

wait

cat ss-*.txt > ss.txt
