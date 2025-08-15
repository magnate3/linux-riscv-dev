#!/bin/bash

# This is a script that compiles individual unit tests and runs them and
# compares them with the expected outputs to check if they failed or passed.
#
# Input file	: the file that is passed as argument to each unit (I)
# Compare file	: the file that consists of expected result to Input file (C)
# Output file	: the file that is produced by the unit and `diff`ed to check
# 		  for failure (O)
#
# File Naming format:
#	Input file-> replace .c with .inp
#	Compare file-> replace .c with .cmp
#	Output file-> replace .c with .out (output file is auto generated donot
#	do any changes or create a file with same name)
#
# File Formats (This is the format of each line of the files)
# ============
# 1) test-read-lctrie.c
#    I: xid (space) prefix-length (space) xid
#    C: xid (space) prefix-length (space) xid
#
# 2) test-compare-xids.c
#    I: xid (space) xid
#    C: xid
#
# 3) test-qsort.c
#    I: xid
#    C: xid
#
# 4) test-build-nexthop-table.c
#    I: xid (space) prefix-length (space) xid
#    C: xid
#
# 5) test-entries.c
#    I: xid (space) prefix-length (space) xid
#    C: xid (space) prefix-length (space) xid
#
# 6) test-bitwise-shift-left.c
#    I: xid (space) shifts
#    C: xid
#
# 7) test-bitwise-shift-right.c
#    I: xid (space) shifts
#    C: xid
#
# 8) test-extract.c
#    I: xid (space) length (space) pos
#    C: xid
#
# 9) test-isprefix.c
#    I: Two route tables i.e two input files same as test-read-lctrie.c of same
#    number of entries
#    C: Binary value
#
# 10) test-binsearch.c
#    I: Two files with first file containing sorted (ascending) xids in each
#    line and second file serving as xids to be searched in the first file
#    C: Integers in the range [-1,n-1] 
#	where n->number of xids in first file
#	[0,n-1] indicates search is successful and the index -1 for not found
#
# 11) test-build-patricia.c
#    I: xid (space) length
#    C: Each node is represented with a level where root is level 0 and the
#    values of skip and pointer to base vector, where -1 signifies an internal
#    node.


declare -a arr=("test-read-lctrie" "test-compare-xids" "test-qsort"
"test-build-nexthop-table" "test-entries" "test-bitwise-shift-left"
"test-bitwise-shift-right" "test-extract" "test-build-patricia")

for i in "${arr[@]}"
do
	gcc -w "$i".c -o "$i"
done

for i in "${arr[@]}"
do
	./"$i" "$i".inp > "$i".out
	if diff "$i".out "$i".cmp > /dev/null;
	then
		echo "$i" Passed
	else
		rm "$i".out
		echo "$i" Failed
	fi
done
