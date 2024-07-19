# #! /usr/bin/env bash

# outdir=${SDE_INSTALL}/patch_panel.tofino

# # Don't necessarily need P4 runtime files. If you do, you may want to consider
# # --p4runtime-force-std-externs (forces use of standard extern messages,
# # although i'm not altogether sure what this means). Target and arch are
# # optional if compiling for tofino/tna.
# bf-p4c --target=tofino --arch=tna \
#     --p4runtime-files=${outdir}/patch_panel.p4info.pb.txt \
#     -I. patch_panel.p4 -o ${outdir}

# # Enables use of shorthand `-p` in tofino model and switchd launches
# cp ${SDE_INSTALL}/patch_panel.tofino/patch_panel.conf \
#     ${SDE_INSTALL}/share/p4/targets/tofino/


#! /usr/bin/env bash

set -e -o pipefail

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p ${script_dir}/build;
cd ${script_dir}/build

cmake ${SDE}/p4studio \
    -DCMAKE_INSTALL_PREFIX=${SDE_INSTALL} \
    -DCMAKE_MODULE_PATH=${SDE}/cmake \
    -DP4_NAME=patch_panel \
    -DP4_PATH=$(realpath ${script_dir}/patch_panel.p4)

make && make install