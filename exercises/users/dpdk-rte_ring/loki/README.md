
support Multi-producer multi-consumer queue   

#  make 
```
make
gcc -I. -std=gnu11 -march=core2 -DLOKI_CPU_RELAX_INSTR_PAUSE -O2 -o loki/queue.o -c loki/queue.c
gcc -I. -std=gnu11 -march=core2 -DLOKI_CPU_RELAX_INSTR_PAUSE -O2 -o loki/debug.o -c loki/debug.c
gcc -I. -std=gnu11 -march=core2 -DLOKI_CPU_RELAX_INSTR_PAUSE -O2  -o tests/queue-test.test tests/queue-test.c loki/queue.o loki/debug.o -lpthread
root@ubuntux86:# ls
loki  Makefile  tests
```

#  run

```
root@ubuntux86:# ./tests/queue-test.test 
Usage: ./tests/queue-test.test <queue-size> <producer-count> <consumer-count> <push-len> <pop-len>
root@ubuntux86:# ./tests/queue-test.test  1024 4 4 2 2
Producer n=255 starting from 1, block of len 2
Producer n=256 starting from 256, block of len 2
Producer n=256 starting from 512, block of len 2
Producer n=256 starting from 768, block of len 2
Consumer, block of len 2
Consumer, block of len 2
Consumer, block of len 2
Consumer, block of len 2
Waiting for the producers
Producer 0 done
Producer 1 done
Producer 2 done
Producer 3 done
Signal the consumers to exit
Waiting for the consumers
Consumer 0 done
Consumer 1 done
Consumer 2 done
Consumer 3 done
OK
```

#  atomic_load_n