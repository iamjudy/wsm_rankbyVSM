
[]

while result < 10:
    if relevanceType == 'Cosine Similarity':
        _max = max(scores)
        i = scores.index(_max)
        try:
            print(doc_name[i], _max, sep='\t')
        except:
            print(i, _max)
        scores[i] = 0
        result+=1
    else:
        _min = min(scores)
        i = scores.index(_min)
        try:
            print(doc_name[i], _min, sep='\t')
        except:
            print(i, _min)
        scores[i] = 0
        result+=1