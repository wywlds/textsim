import common
import re
vocab,idx=common.getVocab()
def findWord(word):
    return str(vocab.index(word))
def simple_preprocess(sent):
    words = re.split(",| ", sent)
    ids=[]
    for word in words:
        word = word.lower()
        if word == "":
            continue
        else:
            if word.endswith("'s"):
                #print word
                ids.append(findWord(word[:len(word)-2]))
                ids.append(findWord("'s"))
            elif word.endswith("n't"):
                #print word
                ids.append(findWord(word[:len(word)-3]))
                ids.append(findWord("not"))
            elif word.endswith("."):
                ids.append(findWord(word[:len(word)-1]))
            else:
                ids.append(findWord(word))
    return ids

def convert(input, output):
    fin = open(input)
    fout = open(output, "w")
    for l in fin.readlines():
        items = l.split("\t")
        sent1 = items[1]
        sent2 = items[2]
        score = "%.1f" % float(items[3])
        ids1=simple_preprocess(sent1)
        ids2=simple_preprocess(sent2)
        fout.write("\t".join((score, " ".join(ids1), " ".join(ids2))))
        fout.write("\n")
    fout.close()

if __name__=="__main__":
    convert("../dataset/sick/train.txt", "../trainformat2.txt")
    convert("../dataset/sick/test.txt", "../testformat2.txt")