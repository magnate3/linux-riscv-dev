#! /bin/bash
i=10
while [[ $i -gt 0 ]] ; do
#	echo "create tags file...\n"
	ctags -R .
	sleep 5
done;
