import ldamodel
import ldapredict
if __name__=="__main__":
    for num_topics in range(10,210,10):
        print num_topics
        ldamodel.lda(num_topics)
        ldapredict.predict(num_topics)