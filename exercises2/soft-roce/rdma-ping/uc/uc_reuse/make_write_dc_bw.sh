#/bin/sh
rm ib_write_dc_bw
rm src/write_dc_bw.o
gcc -DHAVE_CONFIG_H -I.     -g -Wall -D_GNU_SOURCE -O3 -g -O2 -MT src/write_dc_bw.o -MD -MP -MF $depbase.Tpo -c -o src/write_dc_bw.o src/write_dc_bw.c &&\
	mv -f $depbase.Tpo $depbase.Po
/bin/bash ./libtool  --tag=CC   --mode=link gcc  -g -Wall -D_GNU_SOURCE -O3 -g -O2   -o ib_write_dc_bw src/write_dc_bw.o libperftest.a -lm  -lmlx5  -lrdmacm -libverbs  -lpci -lpthread
