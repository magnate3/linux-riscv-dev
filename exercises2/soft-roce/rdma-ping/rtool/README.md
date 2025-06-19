rdma_resource_lat
===========

An tool to monitor RDMA PD, MR, ucontext registration latency in real time.

Homepage: https://github.com/paravpandit/rtool/

Please send your patches, issues and questions to
https://github.com/paravpandit/rtool/issues/

Supported OS
------------

* GNU/Linux

Packages
--------

None at this time.

Examples
--------

Show RDMA MR registration latency

```
$ rdma_resource_lat -d mlx5_0 -s 4G
```

Show RDMA MR registration latency using huge pages

```
$ rdma_resource_lat -d mlx5_0 -s 1G -u

```

Show RDMA PD registration latency

```
$ rdma_resource_lat -d mlx5_0 -R pd

```


Authors
-------

* Parav Pandit <parav@nvidia.com>

Licensed under GPLv3 (or later) <http://www.gnu.org/licenses/gpl-3.0.txt>
