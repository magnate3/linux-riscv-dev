def get_bw(file_name, bw_mr): # 0 for bw, 1 for message rate
    res = []
    next_active = 0
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            line_list = line.split(' ')
            # print("line_list[0]: " + line_list[0])
            if line_list[0] == "#bytes":
                next_active = 1
            if next_active == 1:
                if len(line_list[-2]) > 8:
                    mrate = line_list[-2][0:7]
                else:
                    mrate = line_list[-2]
                line_list[-1].strip()
                # print("res for every line: " + line_list[-1])
                res.append(float(line_list[-1]))
    if next_active == 0:
        raise Exception("no output!")
    if len(res) == 0:
        return 0
    else:
        return (sum(res) / len(res))

def get_message_rate(file_name):
    res = []
    next_active = 0
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            line_list = line.split(' ')
            # print("line_list[0]: " + line_list[0])
            if line_list[0] == "#bytes":
                next_active = 1
            if next_active == 1:
                if len(line_list[-1]) > 8:
                    mrate = line_list[-1][0:7]
                else:
                    mrate = line_list[-1]
                line_list[-1].strip()
                # print("res for every line: " + line_list[-1])
                res.append(float(line_list[-1]))
    if next_active == 0:
        raise Exception("no output!")
    if len(res) == 0:
        return 0
    else:
        return (sum(res) / len(res))