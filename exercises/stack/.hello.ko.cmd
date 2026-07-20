cmd_/root/SimplestLKM/hello.ko := ld -EL -r -maarch64linux -T ./scripts/module-common.lds --build-id  -o /root/SimplestLKM/hello.ko /root/SimplestLKM/hello.o /root/SimplestLKM/hello.mod.o ;  true
