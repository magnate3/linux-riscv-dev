all: pg

pg:
	gcc random_number.c -pg -g -o ran.o -lm
	gcc -pg -g -c timing.c
	gcc -pg -g -c Event.c
	gcc -pg -g -c OptimizationSplay.c
	gcc -pg -g -c memory.c
	gcc markov_hold_model.c -pg -g -o mar.o -lm
clear: 
	rm *.o
