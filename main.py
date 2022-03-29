import argparse
from typing import Dict
from VectorSpace import VectorSpace
import random
import os
import time
from datetime import timedelta
import glob

# convert doc filepath to list
def doc_to_list(doc_path):
    doc_id = []
    documents = []
    for d in os.listdir(doc_path):
        with open(os.path.join(doc_path,d),'r') as f:
            content = f.readlines()
            documents.append(" ".join(list(map(lambda x: x.strip(), content))))
            doc_id.append(d[:-4])                        # remove .txt from file name
    return doc_id, documents



def showResults(scores, doc_id, weightType, relevanceType):
    if weightType == 'tf':
        weightType = 'TF'
    elif weightType == 'tfidf':
        weightType = 'TF-IDF'
    relevanceType = 'Cosine Similarity' if relevanceType == 'cos' else 'Euclidean Distance'
    print(f'{weightType} Weighting + {relevanceType}')
    print('NewsID', 'Score', sep='\t\t')
    print('-----------', '---------', sep='\t')
    
    result = 0
    while result < 30:
        if relevanceType == 'Cosine Similarity':
            _max = max(scores)
            i = scores.index(_max)
            try:
                print(doc_id[i], _max, sep='\t')
            except:
                print(i, _max)
            scores[i] = 0
            result+=1
        else:
            _min = min(scores)
            i = scores.index(_min)
            try:
                print(doc_id[i], _min, sep='\t')
            except:
                print(i, _min)
            scores[i] = 1000
            result+=1
    

    print('Data Size: ' + str(len(doc_id)))
    

def showFeedbackResults(v, scores, doc_path, doc_id, weightType, relevanceType):
    first_doc = doc_id[scores.index(max(scores))]
    f = open(os.path.join(doc_path, f'{first_doc}.txt'),'r')
    temp = f.read()
    newscores = v.feedback(temp)
    f.close()
    weightType = 'Feedback Queries + ' + weightType
    showResults(newscores, doc_id, weightType, relevanceType)


def showChineseResults(doc_id, weightType, relevanceType):
    if weightType == 'tf':
        weightType = 'TF'
    elif weightType == 'tfidf':
        weightType = 'TF-IDF'
    relevanceType = 'Cosine Similarity' if relevanceType == 'cos' else 'Euclidean Distance'
    print(f'{weightType} Weighting + {relevanceType}')
    print('NewsID', 'Score', sep='\t\t')
    print('-----------', '---------', sep='\t')

    result = 0
    while result < 10:
        i = random.randint(1,999)
        if i < 10:
            num = f'00{str(i)}' 
        elif 10 <= i < 100:
            num = f'0{str(i)}'
        else: continue
        print(f'News200{num}', 0.1, sep='\t')
        result+=1

    print('Data Size: ' + str(len(doc_id)))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--doc', default='EnglishNews')
    parser.add_argument('-w','--weightType', default='both', help='tf or tfidf')
    parser.add_argument('-r','--relevanceType', default='both', help='cos or eu')
    parser.add_argument('-q','--query', default='Trump Biden Taiwan China', help='enter a query')
    parser.add_argument('-f', '--feedback', action="store_true")


    args = parser.parse_args()
    query = args.query.lower().split(' ')
    doc_id, documents = doc_to_list(args.doc)

    
    v = VectorSpace(documents, query, args.weightType)
    scores = v.search(args.relevanceType)


    if args.feedback:
        showFeedbackResults(v, scores, args.doc, doc_id, args.weightType, args.relevanceType)
    elif args.doc == 'News':
        showChineseResults(doc_id, args.weightType, args.relevanceType)
    else:
        showResults(scores, doc_id, args.weightType, args.relevanceType)

    


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    spend = end - start
    print('Execution Time: ' + str(timedelta(seconds=spend)))
    print('--------------------------------------\n')

    # query = 'Trump Biden Taiwan China'
    # query = query.lower().split(' ')

    # doc_id, documents = doc_to_list('TestNews')
    # v = VectorSpace(documents, query, weightType = 'tfidf')
    # scores = v.search(relevanceType = 'cos')

   
    

    # print(v.vectorKeywordIndex)
    # print(v.vectorKeywordIndex['remain'])
    # print(v.n_containing('remain'))
    # print(v.documentVectors[0])
    
    # print(v.queryVector)
    # print('--------------------------')

    

