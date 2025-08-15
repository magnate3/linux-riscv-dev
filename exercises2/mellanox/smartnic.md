## Intro Docs ##

* [DPU-Based Hardware Acceleration: A Software Perspective](https://www.linleygroup.com/uploads/nvidia-doca-white-paper.pdf)

## Setup ##

* [BlueField-2 Quickstart Guide for Clemson R7525s](https://groups.google.com/g/cloudlab-users/c/Xk7F46PpxJo)
* [NVIDIA Mellanox Bluefield-2 SmartNIC Hands-On Tutorial](https://medium.com/codex/getting-your-hands-dirty-with-mellanox-bluefield-2-dpus-deployed-in-cloudlabs-clemson-facility-bcb4e689c7e6): Host setup is little outdate. See `Host setup` bellow.
* [NVIDIA Mellanox Bluefield-2 SmartNIC Hands-On Tutorial: “Rig for Dive” — Part III: Ultimate Cloudlab Setup](https://medium.com/codex/nvidia-mellanox-bluefield-2-smartnic-dpdk-rig-for-dive-part-iii-ultimate-cloudlab-setup-7efd8b47a480)
* [NVIDIA Mellanox Bluefield-2 SmartNIC Hands-On Tutorial: “Rig for Dive” — Part V: Install the Latest Bluefield OS with DPDK and DOCA](https://medium.com/codex/nvidia-mellanox-bluefield-2-smartnic-hands-on-tutorial-rig-for-dive-65a6b7278b23)
* [NVIDIA BlueField-2 Ethernet DPU User Guide](https://docs.nvidia.com/networking/display/BlueField2DPUENUG/NVIDIA+BlueField-2+Ethernet+DPU+User+Guide)
* [Host setup for bluefield DPU](https://docs.nvidia.com/networking/display/BlueFieldDPUOSLatest/Updating+Repo+Package+on+Host+Side)
* [Installing Ubuntu on BF2](https://docs.nvidia.com/networking/display/BlueFieldDPUOSLatest/Deploying+BlueField+Software+Using+BFB+from+Host)
* [Download and install Mellanox Technologies GPG-KEY for apt](https://docs.nvidia.com/networking/display/BlueFieldDPUOSLatest/Updating+DPU+Software+Packages+Using+Standard+Linux+Tools): This is needed to install packages on DPU.
* [Configuring NVIDIA BlueField2 SmartNIC](https://insujang.github.io/2022-01-06/configuring-nvidia-bluefield2-smartnic/)
* [BF2 Hardware Installation](https://docs.nvidia.com/networking/display/BlueField2DPUVPI)
* [Troubleshooting BF2 Interfaces](https://docs.nvidia.com/networking/display/BlueFieldDPUOSLatest/Host-side+Interface+Configuration)
* [Installation and Initialization](https://docs.nvidia.com/networking/m/view-rendered-page.action?abstractPageId=15042650): Troubleshooting
* [BF SmartNIC modes](https://mellanox.my.site.com/mellanoxcommunity/s/article/BlueField-SmartNIC-Modes)

## DPU Docs ##

* [NVIDIA BLUEFIELD DPU PLATFORM OPERATING SYSTEM v3.9.2 DOCUMENTATION](https://docs.nvidia.com/networking/display/BlueFieldDPUOSLatest)
* [NVIDIA BLUEFIELD-2 BMC SOFTWARE USER MANUAL v2.8.2](https://docs.nvidia.com/networking/display/BlueFieldBMCv282)

## DPU Programming ##

* [BlueField 2 DOCA SDK Index Page](https://developer.nvidia.com/networking/doca): Ubuntu bfb image at the end of the page.
* [DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/)
* [DOCA programming guide overview](https://docs.nvidia.com/doca/sdk/pdf/programming-guides-overview.pdf)
* [DOCA Core programming guide](https://docs.nvidia.com/doca/sdk/doca-core-programming-guide/index.html)
* [DOCA DMA programming guide [PDF]](https://docs.nvidia.com/doca/sdk/pdf/dma-programming-guide.pdf)
* [DOCA Library API [PDF]](https://docs.nvidia.com/doca/sdk/pdf/doca-libraries-api.pdf)
* [DOCA DMA sample application](https://docs.nvidia.com/doca/sdk/dma-samples/index.html)
* [DOCA Sample Applications](https://gitlab.com/nvidia/networking/bluefield/doca-sample-applications)
* [Introduction to Developing Applications with NVIDIA DOCA on BlueField DPUs](https://www.nvidia.com/en-us/on-demand/session/other2022-dc0511/)
* [How to deal with dma on the host to access DPU’s memory?](https://forums.developer.nvidia.com/t/how-to-deal-with-dma-on-the-host-to-access-dpus-memory/221441)
* [Developing Applications with NVIDIA BlueField DPU and DPDK](https://developer.nvidia.com/blog/developing-applications-with-nvidia-bluefield-dpu-and-dpdk/)

## Internals ##

* [Scalable Functions](https://github.com/Mellanox/scalablefunctions/wiki)
* [Introduction to switchdev SR-IOV offloads](https://legacy.netdevconf.info/1.2/slides/oct6/04_gerlitz_efraim_introduction_to_switchdev_sriov_offloads.pdf)

## OVS and Hardware offloaded packet processing ##
* [DPU Operation Guide](https://docs.nvidia.com/networking/display/BlueFieldDPUOSLatest/DPU+Operation)
* [NVIDIA DOCA vSwitch and Representors Model](https://docs.nvidia.com/doca/sdk/vswitch-and-representors-model/index.html)
* [Open vSwitch in NVIDIA BlueField SmartNIC](https://insujang.github.io/2022-01-17/open-vswitch-in-nvidia-bluefield-smartnic/)
* [OVS Offload Using ASAP² Direct](https://docs.nvidia.com/networking/pages/viewpage.action?pageId=39285091)
* [Accelerated Receive Flow Steering(aRFS)](https://support.mellanox.com/s/article/howto-configure-arfs-on-connectx-4)
* [Mellanox ASAP DOC](https://network.nvidia.com/sites/default/files/doc-2020/sb-asap2.pdf)
* [Hardware offloads past, present and future](https://www.openvswitch.org/support/ovscon2019/day2/0951-hw_offload_ovs_con_19-Oz-Mellanox.pdf)
* [[VIDEO] Hardware offloads past, present and future](https://www.youtube.com/watch?v=r4Vye71VvKA)
* [Offloading Network Traffic Classification](https://bootlin.com/pub/conferences/2019/elce/chevallier-network-classification-offload/chevallier-network-classification-offload.pdf)
* [OVS Bridge and Open vSwitch (OVS) Basics](https://network-insight.net/2015/11/21/open-vswitch-ovs-basics/)
* [Linux kernel documentation: Network Function Representors](https://docs.kernel.org/next/networking/representors.html)
* [Openflow in a day](https://archive.nanog.org/sites/default/files/mon.tutorial.wallace.openflow.31.pdf)
* [OpenFlow Switch Specification](https://opennetworking.org/wp-content/uploads/2013/04/openflow-spec-v1.3.1.pdf)
* [OpenFlow rules interactions: Definition and detection](https://pdfs.semanticscholar.org/4560/3e947edc0681ceb0d052b918b2412f24fe7f.pdf)
* [Flow Hardware offload with Linux TC flower (OVS)](https://docs.openvswitch.org/en/latest/howto/tc-offload/)
* [TC Hardware offload by Simon Horman](https://legacy.netdevconf.info/2.2/papers/horman-tcflower-talk.pdf)
* [OvS Hardware Offload with TC Flower](http://www.openvswitch.org/support/ovscon2017/horman.pdf)
* [Open vSwitch Offload](https://open-nfp.org/static/pdfs/dxdd-europe-ovs-offload-with-tc-flower.pdf)
* [Configuring OVS-DPDK Offload with BlueField-2](https://support.mellanox.com/s/article/Configuring-OVS-DPDK-Offload-with-BlueField-2)
* [Open vSwitch with DPDK](https://discourse.ubuntu.com/t/open-vswitch-with-dpdk/13085)
* [HW offload performance with TC vs. DPDK (on SmartNIC)](https://www.mail-archive.com/ovs-discuss@openvswitch.org/msg08054.html)
* [OVS Deep Dive 0: Overview](https://arthurchiao.art/blog/ovs-deep-dive-0-overview/)
* [Open vSwitch command reference](https://docs.pica8.com/display/PicOS431sp/PICOS+Open+vSwitch+Command+Reference)
* [OVS Short Command Cheat](https://www.southampton.ac.uk/~drn1e09/ofertie/openvswitch_useful_commands.txt)
* [OpenFlow command examples by NVIDIA](https://docs.nvidia.com/networking/display/Onyxv3102202/OpenFlow+Commands)
* [`ovs-fields` - protocol header fields in OpenFlow and Open vSwitch](https://man7.org/linux/man-pages/man7/ovs-fields.7.html)
* [`ovs-actions` - OpenFlow actions and instructions with Open vSwitch extensions](https://man7.org/linux/man-pages/man7/ovs-actions.7.html)
* [Improve multicore scaling in Open vSwitch DPDK](https://developers.redhat.com/articles/2021/11/19/improve-multicore-scaling-open-vswitch-dpdk)

## NVMe-oF offload ##

* [NVIDIA Reported storage offload performance](https://blogs.nvidia.com/blog/2021/12/21/bluefield-dpu-world-record-performance/)
* [HowTo Configure NVMe over Fabrics](https://support.mellanox.com/s/article/howto-configure-nvme-over-fabrics)
* [HowTo Configure NVMe over Fabrics (NVMe-oF) Target Offload](https://mymellanox.force.com/mellanoxcommunity/s/article/howto-configure-nvme-over-fabrics--nvme-of--target-offload)
* [Simple NVMe-oF Target Offload Benchmark](https://support.mellanox.com/s/article/simple-nvme-of-target-offload-benchmark)
* [NVMeOF BF Docs Index page](https://docs.nvidia.com/networking/display/MLNXENv451010/NVME-oF+-+NVM+Express+over+Fabrics)
* [[PATCH RFC] Introduce verbs API for NVMe-oF target offload](https://www.spinics.net/lists/linux-rdma/msg58512.html)
* [NVME OVER FABRICS OFFLOAD presentation by Mellanox](https://www.openfabrics.org/wp-content/uploads/204_TOved.pdf)
* [Hardware offloads for SPDK](https://ci.spdk.io/download/events/2019-summit/10+SPDK+-+(Mellanox)+Hardware+offloads+for+SPDK.pdf)
* [Setting up Mellanox NVMf offload Issue](https://forums.developer.nvidia.com/t/setting-up-mellanox-nvmf-offload/206709)

## Misc ##

* [How to find PCI address of an ethernet interface?](https://askubuntu.com/questions/654820/how-to-find-pci-address-of-an-ethernet-interface)
* [PCI Express I/O Virtualization Howto](https://docs.kernel.org/PCI/pci-iov-howto.html)
* [HowTo Change the Ethernet Port Speed of Mellanox Adapters (Linux)](https://support.mellanox.com/s/article/howto-change-the-ethernet-port-speed-of-mellanox-adapters--linux-x)
* [Orchestrate offloaded network functions on DPUs with Red Hat OpenShift](https://developers.redhat.com/articles/2022/04/26/orchestrate-offloaded-network-functions-dpus-red-hat-openshift#)
* [Understanding mlx5 ethtool Counters](https://enterprise-support.nvidia.com/s/article/understanding-mlx5-ethtool-counters)
* [HowTo Configure DPDK Packet Generator for ConnectX-4](https://mellanox.my.site.com/mellanoxcommunity/s/article/howto-configure-dpdk-packet-generator-for-connectx-4)

## Tools ##

* `mst`
* `mlxconfig`
* `mlxfwmanager`
* `devlink`
* `mlxdevm`