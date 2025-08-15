#! /bin/bash
if [ -z $1 ]; then
	echo "usage: $0 <interface> [2nd interface]"
	exit 1
fi

source common_irq_affinity.sh

CORES=$((`cat /proc/cpuinfo | grep processor | tail -1 | awk '{print $3}'`+1))
hop=1
INT1=$1
INT2=$2

if [ -z $INT2 ]; then
	limit_1=$CORES
	echo "---------------------------------------"
	echo "Optimizing IRQs for Single port traffic"
	echo "---------------------------------------"
else
	echo "-------------------------------------"
	echo "Optimizing IRQs for Dual port traffic"
	echo "-------------------------------------"
	limit_1=$((CORES/2))
	limit_2=$CORES
	IRQS_2=$( get_irq_list $INT2 )
fi

IRQS_1=$( get_irq_list $INT1 )

if [ -z "$IRQS_1" ] ; then
	echo No IRQs found for $1.
else
	echo Discovered irqs for $1: $IRQS_1
	core_id=0
	for IRQ in $IRQS_1
	do
		if is_affinity_hint_set $IRQ ; then
			affinity=$(cat /proc/irq/$IRQ/affinity_hint)
			set_irq_affinity $IRQ $affinity
			echo Assign irq $IRQ to its affinity_hint $affinity
		else
			echo Assign irq $IRQ core_id $core_id
			affinity=$( core_to_affinity $core_id )
			set_irq_affinity $IRQ $affinity
			core_id=$(( core_id + $hop ))
			if [ $core_id -ge $limit_1 ] ; then core_id=0; fi
		fi
	done
fi

echo 

if [ "$INT2" != "" ]; then
	IRQS_2=$( get_irq_list $INT2 )
	if [ -z "$IRQS_2" ]; then
		echo No IRQs found for $2.
	else
		echo Discovered irqs for $2: $IRQS_2
		core_id=$limit_1
		for IRQ in $IRQS_2
		do
			if is_affinity_hint_set $IRQ ; then
				affinity=$(cat /proc/irq/$IRQ/affinity_hint)
				set_irq_affinity $IRQ $affinity
				echo Assign irq $IRQ to its affinity_hint $affinity
			else
				echo Assign irq $IRQ core_id $core_id
				affinity=$( core_to_affinity $core_id )
				set_irq_affinity $IRQ $affinity
				core_id=$(( core_id + $hop ))
				if [ $core_id -ge $limit_2 ] ; then core_id=$limit_1; fi
			fi
		done
	fi
fi
echo
echo done.

