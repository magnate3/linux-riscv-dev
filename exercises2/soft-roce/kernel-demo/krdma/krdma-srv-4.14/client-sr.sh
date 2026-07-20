#!/bin/bash

make clean
make

insmod krdma.ko server=0 rw=0

