i = 0
with open('D:/Project_GD2/data.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if i == 5: break
        print(line.strip())
        i += 1
