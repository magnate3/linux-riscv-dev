#!/bin/sh -e
# SPDX-License-Identifier: GPL-2.0

case "$1" in
	build)
		echo Making ...
		make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
		echo Completed.
		;;
	clean)
		echo Cleaning ...
		make -C /lib/modules/$(uname -r)/build M=$(pwd) clean
		echo Completed.
		;;
	install)
		echo "Installing sblkdev"
		SBLKDEV_PATH=/lib/modules/$(uname -r)/kernel/drivers/block
		mkdir -p ${SBLKDEV_PATH}
		cp sblkdev.ko ${SBLKDEV_PATH}/
		depmod
		echo Completed.
		;;
	uninstall)
		echo "Uninstalling sblkdev"
		SBLKDEV_PATH=/lib/modules/$(uname -r)/kernel/drivers/block
		rm -f ${SBLKDEV_PATH}/sblkdev.ko
		depmod
		;;
	*)
		echo "Usage "
		echo "Compile project: "
		echo "	$0 {build | clean} "
		echo "Install module : "
		echo "	$0 {install | uninstall}"
		exit 1
esac
