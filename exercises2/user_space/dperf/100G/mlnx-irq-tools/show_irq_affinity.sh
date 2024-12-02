#! /bin/bash
if [ -z $1 ]; then
        echo "usage: $0 <interface> "
        exit 1
fi

source common_irq_affinity.sh

IRQS=$( get_irq_list $1 )

for irq in $IRQS
do
	show_irq_affinity $irq
done

