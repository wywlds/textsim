import re
from matplotlib import pyplot as plt
if __name__=="__main__":
    fd = open("./evaluation_result")
    num_topics=10
    x=[]
    r=[]
    rho=[]
    mse=[]
    for l in fd.readlines():
        items = re.findall("[-+]?\d+[.]?\d*e?[-+]?\d*", l)
        x.append(num_topics)
        num_topics += 10
        r.append(items[0])
        rho.append(items[2])
        mse.append(items[4])
    plt.plot(x, r, label="pearson")
    plt.plot(x, rho, label="spearson")
    plt.plot(x, mse, label="mse")
    plt.legend()
    plt.xlabel("num_topics")
    plt.ylabel("metrics")
    plt.show()
