# Homework 3

**Advanced Operating Systems, Term 1, 2022** <br/>
**Updated**: Tuesday 4.1.2022 at 17:41 IDT <br/>
**Due**: Tuesday, 25.1.2022 at 11:59pm IDT

## Instructions

Group collaboration (up to 2 people) is permitted for the Group Programming
section. All other problems in this assignment are to be done individually.
Submission of all homework is recommended to be made via Git.

## Individual Problems:

(1) Explain, in your own words, the mechanisms depicted in the diagrams in
slides 26 and 27 of Lecture #9.

(2) When the OS runs out of memory it resorts to swapping pages to storage.
Consider a system with QEMU/KVM hypervisor running several guests:

* (a) What happens if the guest runs out of memory and starts swapping? How
  does it affect the host? Describe in (low-level) detail what happens when
  a guest-userspace process tries to access a guest-swapped page).

* (b) What happens when the host runs out of the memory, and swaps out pages
  that are used by the hypervisor? How does it affect the guest? Describe in
  (low-level) details what happens when a guest-userspace process tries to
  access a host-swapped page.

(3) One difference between plain virtualization and nested virtualization is
that the former can leverage EPT/NPT hardware extensions, while the latter
cannot do so directly.

* (a) What are the pros and cons of adding another EPT/NPT layer for nested
  virtualization?

* (b) In the _Turtles Project_, nested virtalization uses "EPT compression"
  to optimize memory virtualization. To some extent, this technique reminds
  of the optimization used in the _L3 µ-kernel_ to reduce TLB flushed. Explain
  the common theme to both. In a hindsight, could the _L3 µ-kernel_ benefit
  from today's EPT? If so, how?

(4) The _Antfarm_ project relied on shadow page-tables for introspection, to
learn about processes inside the guest. It was built before the introduction
of EPT/NPT. How would the use of EPT affect the techniques describes in the
paper? What changes (if any) would you propose to adapt their system to EPT,
and how would such changes affect its performance?

## Group programming

The group programming will be done using a Linux VM. Linux platforms can run
on many different architectures, but we will use the X86(64) CPU family. We
will develop on the Linux 5.4 kernel.

### (1) KVM: hypervisor, guest(s), and hypercalls

In this assignment you will extend a hypervisor based on KVM, to understand how
hardware virtualization works in practice.

For this assignment, you need to understand the basics of KVM's API; See for
example [this LWN.net article](https://lwn.net/Articles/658511/).
You will also need KVM installed on your system (see instructions for Ubuntu
[in this link](https://help.ubuntu.com/community/KVM/Installation)).

Note: for the programming exercises below, if you run Linux in a VM (i.e. not
natively), then you need to enable nested virtualization in your VM settings.

**(a) Study an example: "Hello, World!"**

Take a look at [KVM "Hello, world!"](https://github.com/dpw/kvm-hello-world).
In this part you will read and understand the code, and answer the following
questions. (You may insert `printf()` statements to get some answers).

(NOTE: the code doesn't build properly as is; Look at
[issue 15](https://github.com/dpw/kvm-hello-world/issues/15)
for the solution, or clone from a
[fixed fork of the code](https://github.com/purplewall1206/kvm-hello-world)).

The code in _kvm-hello-world.c_ implements a simple hypervisor, that runs the
code in _guest.c_ as its guest. In order to compile it:

        make kvm-hello-world
        ./kvm-hellow-world -l

(With the above, the hypervisor runs the guest in _"long"_ mode, i.e. 64-bit
compatibility; other modes are possible).

The hypervisor first allocates memory for the guest:

(a.1) What is the size of the guest (physical) memory? How and where in the
code does the hypervisor allocate it? At what host (virtual) address is this
memory mapped?

(a.2) Besides the guest memory, what additional memory is allocated? What is
stored in that memory? Where in the code is this memory allocated? At what
host/guest? (virtual/physical?) address is it located?

The hypervisor then formats the guest memory and registers, to prepare for its
execution. (From here on, assume _"long"_ mode).

(a.3) The guest memory area is setup to contain the guest code, the guest page
table, and a stack. For each of these, identify where in the code it is setup,
and the address range it occupies (both guest-physical and host-virtual).

(a.4) Examine the guest page table. How many levels does it have? How many 
pages does it occupy? Describe the guest virtual-to-physical mappings: what
part(s) of the guest virtual address space is mapped, and to where?

For both (a.3) and (a.4), illustrate/visualize the hypervisor memory layout
and the guest page table structure and address translation. (Preferably in
text form).

(a.5) At what (guest virtual) address does the guest start execution? Where is
this address configured?

Next, the hypervisor proceeds to run the guest. For simplicity, the guest code
(in _guest.c_) is very basic: one executable, a simple page table and a stack
(no switching). It prints "Hello, world!" using the `outb` instruction (writes
a byte to an IO port) which is a protected instruction that causes the guest to
exit to the hypervisor, which in turn prints the character.

(Notice how the `outb` instruction is embedded as assembly inside the guest's
C code; See [here](https://wiki.osdev.org/Inline_Assembly) for details on how
inline assembly works).

After the guest exits, the hypervisor regains control and checks the reason for
the exit to handle it accordingly.

(a.6) What port number does the guest use? How can the hypervisor know the port
number, and read the value written? Which memory buffer is used for this value?
How many exits occur during this print?

Finally, the guest writes the number 42 to memory and EAX register, and then
executes the `hlt` instruction.

(a.7) At what guest virtual (and physical?) address is the number 42 written?
And how (and where) does the hypervisor read it?

**(b) Extend with new hypercalls**

Implement hypercalls from the guest to the hypervisor, using the `in`/`out`
instructions to pass data between the guest and the hypervisor. (How would the
hypervisor identify the `in` instruction?).

Use this snippet to pass 32-bit integer from the guest to the hypervisor:

        static inline void outb(uint16_t port, uint32_t value) {
            asm("out %0,%1" : /* empty */ : "a" (value), "Nd" (port) : "memory");
        }

Use this snippet to pass 32-bit integer back from the hypervisor to the guest:

        static inline uint32_t inb(uint16_t port) {
            uint32_t ret;
            asm("in %1, %0" : "=a"(ret) : "Nd"(port) : "memory" );
            return ret;
        }

(The inline assembly code stores the port number in the EDX register, invokes
the `in` instruction, and stores its return value in the EAX register; The C
compiler uses the EAX register as the return value for the function).

The 32-bit value passed can be an integer, a pointer, or anything you decide.

(b.1) Write a hypercall that passes a string to the hypervisor to print on the
screen; And a function `void print(const char *str)` (in the guest) that uses
this hypercall. The hypercall must do its work with a single guest exit. You
need to decide what value (and how) to pass to the hypervisor.

(b.2) Write a hypercall that returns the number of guest exits occurred since
it started; And a function `int exits(void)` (in the guest) that uses this
hypercall. Note that the host code does not print the result, but just returns
it to the guest.

You should test both functions from the guest. In particular, you should (at
least) use `exits()` to count the exits incurred by `print()`, and `print()` to
display it.

**(c) Filesystem para-virtualization**

Implement filesystem para-virtualization using new hypercalls to emulate file
system functionality for the guest.

(c.1) Write a hypercall that accepts a pathname and opens (in the host) the
file; And a function `int open(const char *path)` that uses it.

(c.2) Write a hypercall that accepts a \<pointer, len\> and reads from a file
into that buffer in the guest; And a function `int read(void *buf, int len)`
that uses it. (How to pass two parameters via the hypercall?). You may assume
the file was previously opened. 

(c.3) Write a hypercall that accepts a \<pointer, len\> and writes to a file
from that buffer in the guest; And a function `int write(void *buf, int len)`
that uses it. You may assume the file was previously opened. 

(c.4) Write a hypercall that closes the file; And a function `void close()`
that uses it. You may assume the file was previously opened. 

**(d) Bonus: hypercall in KVM**

KVM already defines several hypercalls, together with a mechanism for guests
to invoke them (architecture-dependent). For the x86, this is implemented in
`arch/x86/kvm/x86.c:kvm_emulate_hypercall()`. It is called from code specific
to Intel/AMD when the guest executes the `VMCALL`/`VMMCALL`, respectively.

Unlike the hypercalls from before, which were passed by KVM to the hypervisor
process in userspace, these hypercalls invoke functions inside the KVM kernel
module.

Implement any of the hypercalls from before by extending the KVM kernel module
(in the function above). You will find sample code how to invoke the hypercall
from the guest in _arch/x86/include/asm/kvm_para.h_. Notice the definition of
the macro `KVM_HYPERCALL`, whose purpose is to select the suitable instruction
(`VMCALL`/`VMMCALL`) depending on the CPU type.

For this, you will need to modify and rebuild the KVM kernel module (but not
the entire kernel).

The bonus question is optional.

**(e) Multiple vCPUs**

Extend the program _kvm-hello-world.c_ to launch a guest with 2 vCPUs, with
an instance of the guest running on each vCPU independently and concurrently, 
including invoking hypercalls. You will need to use another process/thread to
run the second vCPU. For this setup, double the size of the guest physical
memory and allow each guest instance (vCPU) to use half, to protect them from
each other. You will need to provide a distinct guest page table for each of
the guest instances (in the _"long"_ mode).

(e.1) Add code or pseudo-code, to _kvm-hello-world_ and to _guest.c_ that
implements the necessary changes described above.

(e.2) Bonus: complete your code such that it compiles and works, and test it
to demonstrate that it operates correctly.

The bonus question is optional.

### (2) Containers and namespaces

In this assignment you will implement a simples container runtime that can
spawn a command in an isolated environment.

For this assignment, you need to understand the basics of Linux namespaces;
Read through __"Digging into Linux namespaces"__:
[part 1](https://blog.quarkslab.com/digging-into-linux-namespaces-part-1.html)
and 
[part 2](https://blog.quarkslab.com/digging-into-linux-namespaces-part-2.html).

We will use the following steps to build a fully isolated environment for a
given process:

1. Create user namespace; remap the UIDs/GIDs in the new _userns_
2. Create uts namespaces; change hostname in the new _utsns_
3. Create ipc namespace
4. Create net namespace; create and configure veth interface pair
5. Create pid namespace
6. Create mnt namespace; mount /proc inside the new _mntns_

(Note that the process would run in an isolated environment, but would share
the same root filesystem as the parent, just in a separate _mntns_).

These steps can be done in userspace as follows:

            Parent shell                     Child shell
            -------------------------------  -----------------------------  
          1                                  # (1) create (privileged) userns
          2   
          3                                  $ unshare -U --kill-child /bin/bash
          4                                  $ echo "my-user-ns" > /proc/$$/comm
          5                                  $ id
          6                                  uid=65534(nobody) gid=65534(nogroup) groups=65534(nogroup)
          7   
          8   
          9   $ ps -e -o pid,comm | grep my-user-ns
         10   22310,my-user-ns?
         11   
         12   $ sudo bash -c 'echo "0 1000 1000" > /proc/22310/uid_map'
         13   $ sudo bash -c 'echo "0 1000 1000" > /proc/22310/gid_map'
         14   
         15                                  $ id
         16                                  uid=0(root) gid=0(root) groups=0(root),65534(nogroup)
         17                                  
         18                                  # (2,3) create utsns and ipcns
         19   
         20                                  $ unshare --ipc --uts --kill-child /bin/bash
         21                                  $ hostname isolated
         22                                  $ hostname
         23                                  isolated
         24   
         25                                  # (4) create netns
         26                                  $ unshare --net --kill-child /bin/bash
         27                                  $ echo "my-net-ns" > /proc/$$/comm
         28   
         29   $ ps -e -o pid,comm | grep my-user-ns
         30   22331,my-net-ns?
         31   
         32   $ sudo ip link add veth0 type veth peer name peer0
         33   $ sudo ip link set veth0 up
         34   $ sudo ip addr add 10.11.12.13/24 dev veth0
         35   
         36   $ sudo ip link set peer0 netns /proc/22331/ns/net
         37   
         38                                  $ ip link
         39                                  1: lo: <LOOPBACK> mtu 65536 qdisc noop state DOWN mode DEFAULT group default qlen 1000
         40                                      link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
         41                                  9: peer0@if10: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
         42                                      link/ether 76:8d:bb:61:1b:f5 brd ff:ff:ff:ff:ff:ff link-netnsid 0
         43                                  $ ip link set lo up
         44                                  $ ip link set peer0 up
         45                                  $ ip addr add 10.11.12.14/24 dev peer0
         46   
         47                                  $ ping -c 1 10.11.12.13
         48                                  PING 10.11.12.13 (10.11.12.13) 56(84) bytes of data.
         49                                  64 bytes from 10.11.12.13: icmp_seq=1 ttl=64 time=0.066 ms
         50   
         52                                  # (5,6) create pidns, mntns
         53                                  $ unshare --pid --mount --fork --kill-child /bin/sh
         54                                  $ mount -t proc proc /proc
         55                                  $ ps

(a) Describe the process hierarchy produced by the sequence of commands in the
"child shell" column. How can it be minimized, and what would the hierarchy
look like?

(b) What would happen if you change the order of namespace creation, e.g. run
`unshare --ipc` first? And what would happen if you defer lines 12-13 until
a later time?

(c) What is the purpose of line 4 and lines 9-10 (and similarly, line 27 and
lines 29-30)? Why are they needed?

(d) Describe how to undo and cleanup the commands above. (Note: there is more
than one way; try to find the minimal way). Make sure there are no resources
left dangling around.

(d) Write a program that would implement the sequence above, whose usage is:

        usage: isolate PROGRAM [ARGS...]

For example, the command:

        isolate ps aux

would execute the command "ps aux" inside an isolated environment.

For this, you may use the skeleton below that uses _clone(2)_ to create new
namespaces:

        #define STACK_SIZE (1024*1024)
        char stack_child[STACK_SIZE];

        int create_namespaces()
        {
            int fds[2];
            int flags;
            pid_t pid;

            pipe(fds);
            
            // recieve signal on child process termination
            flags = SIGCHLD | \
                    CLONE_NEWUSER | CLONE_NEWNET | CLONE_NEWUTS | \
                    CLONE_NEWIPC| CLONE_NEWNS | CLONE_NEWPID;

            // the child stack growns downwards 
            pid = clone(child_func, stack_child + STACK_SIZE, flags, fds);
            if (pid == -1) {
                fprintf(stderr,"clone: %s", strerror(errno));
                exit(1);
            }
            
            setup_userns(pid);
            setup_netns(pid);

            write(c->fd[1], &pid, sizeof(int));
            close(c->fd[1]);
            waitpid(pid, NULL, 0);
        }

        void int child_func(void *args)
        {
            int fds[2] = args;
            pid_t pid;

            read(fds[0], &pid, sizeof(int));
            close(fds[0]);

            setup_mntns();
            setup_utsns();

            write(c->fd[1], &pid, sizeof(int));
            close(c->fd[1]);
            waitpid(pid, NULL, 0);
        }

        void int child_func(void *args)
        {
            int fds[2] = args;
            pid_t pid;

            read(fds[0], &pid, sizeof(int));
            close(fds[0]);

            execvp(...);
        }

Note: you may (and should) use the _system()_ helper to implement the
_setup\_mntns()_ and _setup\_netns()_ functions; but not for the
_setup\_utsns()_ and _setup\_userns()_.

(e) Test your program. Does it require root privileges? If so, then why?
How can it be changed to not require these privileges?
