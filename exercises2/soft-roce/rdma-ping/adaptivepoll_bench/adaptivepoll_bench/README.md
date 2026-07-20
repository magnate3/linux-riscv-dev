receiver node:
setupib 100.100.100.92
./write_bw -F -s 1000 -n 1000 -O 0 -p 8888 -I 400

sender node:
setupib 100.100.100.91
./write_bw 100.100.100.92 -F -s 1000 -n 1000 -O 0 -I 400 -p 8888 -o result.txt
