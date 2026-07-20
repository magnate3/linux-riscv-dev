This repository contains work being done to implement the Precision Time Protocol (IEEE 1588) on the Tofino ASIC.

## Requirements

- SDE version >= 9.5.0
- Python >= 3.8
- Thrift >= 0.10.0

## Generate Thrift code for Python

~~~
thrift --gen py $SDE/pkgsrc/bf-drivers/pdfixed_thrift/thrift/ts_pd_rpc.thrift
thrift --gen py $SDE/pkgsrc/bf-drivers/pdfixed_thrift/thrift/res.thrift
~~~
