# simple-kvm-guest

## Quick start

```sh
make -C quick-start/guest
```

```sh
gcc quick-start/kvm.c -o kvm
```

```sh
./kvm
```

```sh
pidstat -p `pidod kvm` 1
```

```sh
08:34:49 PM   UID       PID    %usr %system  %guest    %CPU   CPU  Command
08:34:50 PM     0      3655    0.00    0.00  100.00  100.00     1  kvm
08:34:51 PM     0      3655    0.00    0.00  100.00  100.00     1  kvm
08:34:52 PM     0      3655    0.00    0.00  100.00  100.00     1  kvm
```

## Serial IO

```sh
make -C serial-io/guest
```

```sh
gcc serial-io/kvm.c -o kvm -lrt
```

```sh
./kvm
```
