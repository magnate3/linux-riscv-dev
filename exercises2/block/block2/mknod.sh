#!/bin/bash

ROOT_UID=0
E_NOTROOT=67

if [ "$UID" -ne "$ROOT_UID"  ]
then
     echo "Must be root to run this script."
     exit $E_NOTROOT
fi

if [ -e /dev/mcuosblk ]
then
    echo "Device node is existing now."
else
    echo "Create device node now!"
    mknod /dev/mcuosblk c 221 0
fi


