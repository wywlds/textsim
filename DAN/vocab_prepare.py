import numpy as np
import re
filename = '../glove.6B.300d.txt'
def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab, embd
vocab, embd = loadGloVe(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)

self_vocab={}
def findWordAndAddToVocab(word):
    try:
        if not self_vocab.has_key(word):
            index = vocab.index(word)
            embed = embedding[index]
            self_vocab[word]=embed
    except:
        print "failed word:"+word

def simple_preprocess(sent):
    words = re.split(",| ", sent)
    for word in words:
        word = word.lower()
        if word == "":
            continue
        else:
            if word.endswith("'s"):
                #print word
                findWordAndAddToVocab(word[:len(word)-2])
                findWordAndAddToVocab("'s")
            elif word.endswith("n't"):
                #print word
                findWordAndAddToVocab(word[:len(word)-3])
                findWordAndAddToVocab("not")
            elif word.endswith("."):
                findWordAndAddToVocab(word[:len(word)-1])
            else:
                findWordAndAddToVocab(word)

def processFileName(f):
    testdoc = open(f)
    for l in testdoc.readlines():
        items = l.split("\t")
        sent1 = items[1]
        sent2 = items[2]
        simple_preprocess(sent1)
        simple_preprocess(sent2)

if __name__=="__main__":
    processFileName("../dataset/sick/test.txt")
    processFileName("../dataset/sick/train.txt")
    i = 0
    f=open("../selfvocab.txt","w")
    for (word, embed) in self_vocab.items():
        f.write("\t".join((word, str(i), "\t".join(embed))))
        f.write("\n")
        i += 1
    f.close()