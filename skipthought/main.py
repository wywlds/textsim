import skipthoughts
if __name__=="__main__":
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    testds = open("../dataset/sick/train.txt")
    sents = []
    for l in testds.readlines():
        splits = l.split("\t")
        sents.append(splits[1])
        sents.append(splits[2])
    vectors = encoder.encode(sents)
    vectors.tofile("train.skip")

