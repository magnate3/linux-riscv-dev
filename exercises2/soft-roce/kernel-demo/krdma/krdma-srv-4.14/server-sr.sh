#!/bin/bash

make clean
make

insmod krdma.ko server=1 rw=0

