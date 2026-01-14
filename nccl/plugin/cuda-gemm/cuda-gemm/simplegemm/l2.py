sms = 132
m_blocks = 66
n_blocks = 72

all_blocks = set()
for m in range(m_blocks):
    for n in range(n_blocks):
        all_blocks.add((m, n))

my_blocks = set()
for sm in range(0, 132):
    for bid in range(sm, m_blocks * n_blocks, sms):

        """
        m = sm // 2 #bid % m_blocks
        n = bid // m_blocks + (sm % 2)
        """

        m = (bid // 2) % m_blocks
        n = (bid // 2) // m_blocks * 2 + bid % 2

        print(f"{sm} {m} {n}")
        my_blocks.add((m, n))
        if (m, n) not in all_blocks:
            print(f"egad: {m}, {n}")

for m in range(m_blocks):
    for n in range(n_blocks):
        if (m, n) not in my_blocks:
            print(f"oh no: {m}, {n}")
