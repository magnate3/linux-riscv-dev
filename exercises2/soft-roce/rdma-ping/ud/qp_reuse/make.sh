#/bin/sh
rm ib_send_ud_bw
rm src/send_ud_bw.o
cc -DHAVE_CONFIG_H -I.     -g -Wall -D_GNU_SOURCE -O3 -g -O2 -MT src/send_ud_bw.o -MD -MP -MF $depbase.Tpo -c -o src/send_ud_bw.o src/send_ud_bw.c
/bin/bash ./libtool  --tag=CC   --mode=link gcc  -g -Wall -D_GNU_SOURCE -O3 -g -O2   -o ib_send_ud_bw src/send_ud_bw.o src/multicast_resources.o libperftest.a -libumad -lm  -lmlx5  -lrdmacm -libverbs  -lpci -lpthread
gcc -g -Wall -D_GNU_SOURCE -O3 -g -O2 -o ib_send_ud_bw src/send_ud_bw.o src/multicast_resources.o  libperftest.a -libumad -lm -lmlx5 -lrdmacm -libverbs -lpci -lpthread
depbase=`echo src/send_lat.o | sed 's|[^/]*$|.deps/&|;s|\.o$||'`;\
	gcc -DHAVE_CONFIG_H -I.     -g -Wall -D_GNU_SOURCE -O3 -g -O2 -MT src/send_lat.o -MD -MP -MF $depbase.Tpo -c -o src/send_lat.o src/send_lat.c &&\
	mv -f $depbase.Tpo $depbase.Po
