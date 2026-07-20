for tid in range(0, 128):
    warp = tid // 32
    lane = tid % 32
    row_base = warp * 16
    row_off = lane // 4
    row = row_base + row_off

    col = tid % 4 * 2
    rcs = []
    for n in range(0, 256, 8):
        rcs.extend([
            (row, n + col),
            (row, n + col + 1),
            (row + 8, n + col),
            (row + 8, n + col + 1),
        ])
    print(f"{tid:3d}: " + "  ".join(f"({r:3d}, {c:3d})" for r, c in rcs))
