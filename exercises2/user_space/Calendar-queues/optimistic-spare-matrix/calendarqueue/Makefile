all: serial

serial: 
	gcc -g -c Event.c
	gcc -g -c timing.c
	gcc -g -c VoidSplay.c 
	gcc VoidSplayTest.c -g -o vstest.o
	gcc -g -c Node.c
	gcc -g -c linkedlist.c
	gcc -g -c CalendarQueue.c
	gcc CalendarQueueTest.c -g -o ctest.o
n:
	gcc -c Event.c
	gcc -c timing.c
	gcc -c VoidSplay.c 
	gcc VoidSplayTest.c -g -o vstest.o
	gcc -c Node.c
	gcc -c linkedlist.c
	gcc -c CalendarQueue.c
	gcc CalendarQueueTest.c -o ctest.o
clean: 
	-rm *.o
	-rm out
	-rm *.out
clear: 
	-rm *.o
	-rm out
	-rm *.out
c: 
	-rm *.o
	-rm out
	-rm *.out

