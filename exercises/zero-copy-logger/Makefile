.PHONY: all clean

CFLAGS_EXTRA ?= -ggdb3 -O0 -std=c99 -Wall -Werror -Wextra
IN_EXT =sample_zlog.c
#OUT_EXT=
#OUTS := $(addsuffix $(OUT_EXT), $(basename $(wildcard *$(IN_EXT))))
OUTS=sample_zlog

all:  tag $(OUTS)

up:
	#sudo rmmod zlog || true
	#sudo insmod ./kmod/zlog.ko max_mem_size_order=4
	#sudo ./$(OUTS)
	#sshpass -proot123 scp -o StrictHostKeyChecking=no $(OUTS) root@10.1.1.3:~/
	scp -o StrictHostKeyChecking=no $(OUTS) root@2.2.2.111:~/
	#ssh -o StrictHostKeyChecking=no root@2.2.2.111 ./$(OUTS)

driver:
	cd kmod; make
tag:
	ctags -R

%$(OUT_EXT): %$(IN_EXT)
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -o '$@' '$<'

clean:
	rm -f tags
	rm -f '$(OUTS)'
	rm -f *.o                                          

