# README

Simple character device loadable kernel module (LKM) for Linux.
Compiled module default name is:

<pre>chardev</pre>

The device handles read and write requests to and from the internal buffer.
It also implements simple debugfs and sysfs interfaces.
The device can be opened by one process by default but this can be adjusted through:

<pre>echo &gt; n /sys/class/exercise/exercise_char_dev/exercise_sysfs/max_num_proc</pre>

where 'n' is the number of processes that can have the device open in range <1,10>.

The module exports a debugfs interface under:

<pre>/sys/kernel/debug/exercise_debugfs</pre>

This is a read-only file which will return devices MAJOR number and can be read with:

<pre>cat /sys/kernel/debug/exercise_debugfs</pre>

Reading and writing to the device can be done through this Linux file:

<pre>/dev/exercise_char_dev</pre>

It can be read and written to using cat to read and echo to write:

<pre>echo "test" > /dev/exercise_char_dev
cat /dev/exercise_char_dev</pre>

The device can be also used with standard file operations Linux API (open, read, write, close).

In order to be able to compile the module you need to first install following packages on you system:

<pre>kernel-headers
kernel-devel</pre>

To compile the module simply execute:

<pre>make</pre>

This will use default location of kearnel headers: <pre>/lib/modules/$(shell uname -r)/build</pre>

If the location of kernel headers is non standard you can compile with:

<pre>KDIR=&lt;kernel-headers build folder location&gt; make</pre>

To load the module to the system use:

<pre>insmod chardev.ko</pre>

To unload the module use:

<pre>rmmod chardev</pre>
