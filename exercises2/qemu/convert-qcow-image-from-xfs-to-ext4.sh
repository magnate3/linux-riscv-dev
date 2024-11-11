#!/bin/bash
set -euo pipefail

readonly xfs_device=/dev/nbd0
readonly ext4_device=/dev/nbd1

function log {
    tput bold
    printf "$1\n"
    tput sgr0
}

if [[ $# != 1 ]]; then
    printf "Usage: $(basename $0) <rhel7.qcow2>\n" >&2
    exit 1
fi

readonly xfs_image="$1"
readonly ext4_image="${xfs_image%.qcow2}-ext4.qcow2"
readonly wip_ext4_image="$(dirname $ext4_image)/WIP-$(basename $ext4_image)"

log "Checking prerequisites..."
if [[ ! -f "$xfs_image" ]]; then
    printf -- "$xfs_image file doesn’t exist.\n" >&2
    exit 1
fi

if qemu-img info --output=json "$xfs_image" | jq -e '.format != "qcow2"'; then
    printf -- "$xfs_image file isn’t a QCow2 image.\n" >&2
    exit 1
fi

log "Loading nbd module..."
modprobe nbd max_part=63

log "Creating mount points..."
readonly  xfs_mount=$(mktemp -d)
readonly ext4_mount=$(mktemp -d)

function cleanup {
    log "Cleanup..."
    set +e
    umount -R "$xfs_mount" "$ext4_mount"
    rmdir "$xfs_mount" "$ext4_mount"
    qemu-nbd -d "$xfs_device"
    qemu-nbd -d "$ext4_device"
    rm -f "$wip_ext4_image"
    set -e
}
trap cleanup EXIT

partition_id=p1
xfs_partition=$xfs_device$partition_id
ext4_partition=$ext4_device$partition_id

image_size=$(qemu-img info --output=json "$xfs_image" | jq '."virtual-size"')

log "Mounting the xfs-formated partition..."
qemu-nbd -r -c "$xfs_device" "$xfs_image"
while [[ ! -e "$xfs_partition" ]]; do
    sleep .1
done
mount -o ro "$xfs_partition" "$xfs_mount"

log "Creating the ext4-formated disk..."
qemu-img create -f qcow2 "$wip_ext4_image" "$image_size"
qemu-nbd -c "$ext4_device" "$wip_ext4_image"

log "Formating the ext4-formated disk..."
parted -s "$ext4_device" -- mklabel msdos \
                            mkpart primary ext4 1 -1 \
                            set 1 boot on
mkfs.ext4 "$ext4_partition"

log "Mounting the ext4-formated partition..."
mount "$ext4_partition" "$ext4_mount"

log "Copying source files to destination images..."
cp -a "$xfs_mount"/* "$ext4_mount"

log "Patching files where partition UUID are used"
readonly  xfs_uuid=$(blkid -o value -s UUID "$xfs_partition")
readonly ext4_uuid=$(blkid -o value -s UUID "$ext4_partition")

for file in /etc/fstab \
            /boot/grub/grub.conf \
            /boot/grub2/grub.cfg; do
	sed -i "s/$xfs_uuid/$ext4_uuid/g" "$ext4_mount/$file"
done

log "Disabling SELinux..."
sed -i "s/^SELINUX=.*/SELINUX=disabled/g" "$ext4_mount/etc/selinux/config"

log "Installing GRUB"
for m in /dev /proc /sys /dev/pts; do
    mount --bind "$m" "$ext4_mount$m"
done
chroot "$ext4_mount" /usr/sbin/grub2-install --recheck --target=i386-pc "$ext4_device"

mv "$wip_ext4_image" "$ext4_image"
