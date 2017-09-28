import numpy as np
filename = '../selfvocab.txt'
def loadGloVe(filename):
    vocab = []
    idx = []
    embd = []
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split('\t')
        vocab.append(row[0])
        idx.append(row[1])
        nums=[]
        for num in row[2:]:
            floatnum = float(num)
            nums.append(floatnum)
        embd.append(nums)
    embd.append(np.zeros(300, dtype=float))
    print('Loaded GloVe!')
    file.close()
    return vocab, idx, embd
vocab, idx, embd = loadGloVe(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd,dtype=float)

def getEmb():
    return embedding

def getVocab():
    return vocab,idx

def getDataset(filename):
    fd = open(filename)
    scores = []
    leftsents = []
    rightsents = []
    for l in fd.readlines():
        items = l.split("\t")
        scores.append(float(items[0]))
        leftsent = np.asarray([int(i) for i in items[1].split()])
        rightsent = np.asarray([int(i) for i in items[2].split()])
        leftsents.append(leftsent)
        rightsents.append(rightsent)
    return (leftsents, rightsents, scores)

def getTrainSet():
    return getDataset("../trainformat.txt")

def getTestSet():
    return getDataset("../testformat.txt")

if __name__=="__main__":
    (left, right, scores) = getTrainSet()
    print left[:2]
    print right[:2]
    print scores[:2]
    print left