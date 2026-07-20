# dist-opt
PyTorch distributed optimizer C++ implementation.

## Test

```console
$ DIST_OPT_LOG=1 pytest -s -k test_mp_dist_opt_simple
```

## Known Issues

1. It doesn't work with multiple parameter groups currently;


## Dependency

1. [cameron314/readerwriterqueue](https://github.com/cameron314/readerwriterqueue.git)
2. [pytorch/torch/lib/c10d](https://github.com/pytorch/pytorch/tree/master/torch/lib/c10d)
3. [APEX](https://github.com/NVIDIA/apex)
