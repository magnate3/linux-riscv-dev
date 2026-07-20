addrs = set()

INST_M = 64
for tid in range(128):
    warp = tid // 32
    lane = tid % 32
    base_x1_row = warp * 16
    base_x4_row = base_x1_row + (lane // 8 % 2) * 8
    base_x4_col = lane % 8 + lane // 16 * 8

    base_addr = base_x4_row + INST_M * base_x4_col
    bank = base_addr // 2 % 32

    padded_base_addr = base_x4_row + (INST_M + 8) * base_x4_col
    padded_bank = padded_base_addr // 2 % 32

    swizzle_addr = base_addr ^ ((lane & 7) << 3)
    swizzle_bank = swizzle_addr // 2 % 32

    addrs.add(swizzle_addr)
    print(f"{tid:3d}: ({base_x4_row:3d}, {base_x4_col:3d}): {base_addr:5d} {bank:5d} | {padded_base_addr:5d} {padded_bank:5d} | {swizzle_addr:5d} {swizzle_bank:5d}")

print(len(addrs))
print(sorted(addrs))
for x, y in zip(sorted(addrs), range(0, 1024, 8)):
    print(x, y, "" if x == y else "FAIL")
