
for i in "1" "2" "3" "4"
do
	#ssh node-$i "ps -ef | grep main | grep -v grep | awk '{print $2}' | xargs kill -9"
	ssh node-$i "~/rdmaServer/read/main &" &
done

hostnames="10.10.1.2,10.10.1.3,10.10.1.4,10.10.1.5"
echo $hostnames
./main -h $hostnames -c 1 -n 100000 -m 1024  -M 4
		
