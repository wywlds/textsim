import re
def leven(str1, str2):
    words1=re.split(",| ", str1)
    words2=re.split(",| ", str2)
    len1 = len(words1)
    len2 = len(words2)
    matrix = [0 for n in range(len1*len2)]

    for i in range(len1):
        matrix[i]=i
    for j in range(0, len(matrix), len1):
        if j % len1 == 0:
            matrix[j]=j/len1
    for i in range(1, len1):
        for j in range(1, len2):
            if words1[i-1] == words2[j-1]:
                cost = 0
            else:
                cost = 1
            matrix[j*len1 + i] = min(matrix[(j-1)*len1 + i] + 1,
                                     matrix[j*len1+i-1]+1,
                                     matrix[(j-1)*len1+i-1]+cost)
    return matrix[-1]

if __name__=="__main__":
    print leven("I hava a dream", "I have an dream")