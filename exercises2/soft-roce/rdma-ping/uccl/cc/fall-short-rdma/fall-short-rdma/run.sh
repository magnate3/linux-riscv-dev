#!/bin/sh


scpfiles(){
	for i in {1..5}
	do 
		scp $1 yfle0707@node-$i:~/
	done
}

killprocess(){
	ps -ef | grep $1 | grep -v grep | awk '{print $2}' | xargs kill -9
}

gitpush(){
	com=`date`
	echo $com
	git add *.c *.h *.sh Makefile $1
	git commit -m "$com"
	git push origin master
}
export -f scpfiles
export -f killprocess
export -f gitpush
