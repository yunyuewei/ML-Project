import matplotlib.pyplot as plt
import os
import json

if __name__ == "__main__":

    classes = {}
    with open('garbage_classify_rule.json', 'r') as fi:
        classes = json.load(fi)

    classes = {int(key): value for key, value in classes.items()}
    num_classes = {key: 0 for key in classes.keys()}

    for i in range(42):
        out = os.popen(
            "ls -ar data/{{train,val}}/{} | grep .jpg | wc -l".format(i))
        num = out.read()
        num = num.strip()
        num_classes[i] = int(num)

    num_classes = [(id_c, num_c) for id_c, num_c in num_classes.items()]
    num_classes = sorted(num_classes, key=lambda item: item[1], reverse=True)

    # plt.grid(ls='--')

    fig, ax = plt.subplots(figsize=[10,4])
    plt.bar(list(range(42)), [num for _, num in num_classes])
    plt.xticks(list(range(42)), [index for index, _ in num_classes])
    plt.xlabel("Index of class")
    plt.ylabel("Population")
    plt.tight_layout()
    plt.savefig("dist_dataset.pdf")
    # plt.show()
