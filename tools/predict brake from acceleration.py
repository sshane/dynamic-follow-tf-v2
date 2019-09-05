import ast
d_data = []
with open("D:\Resilio Sync\df\knackerbrot-HOLDEN ASTRA RS-V BK 2017\df-data.67967", "r") as f:
    df = f.read().split("\n")
    for line in df:
        if line != "" and "[" in line and "]" in line and len(line) >= 40:
            d_data.append(ast.literal_eval(line))

for i in d_data:
    if i[0] > 3 and i[6] > 0:
        print(i[1])
        print(i[6])
        print()