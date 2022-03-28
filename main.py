import argparse
from typing import Dict
from VectorSpace import VectorSpace
import sys, getopt
import os
import time
from datetime import timedelta
import glob

# convert doc filepath to list
def doc_to_list(doc_path):
    doc_id = []
    documents = []
    for d in os.listdir(doc_path):
        f = open(os.path.join(doc_path,d),'r')
        content = f.read()
        documents.append(content)
        doc_id.append(d[:-4])                        # remove .txt from file name
        f.close()
    return doc_id, documents

# def doc_to_dict(path):                             #give path of the folder containing all documents
#     dict = {}
#     file_names = glob.glob(path)
#     files_dict = file_names[0:7034]
#     for file in files_dict:
#         name = file.split('/')[-1]
#         with open(file, 'r', errors='ignore') as f:
#             data = f.read()
#         dict[name] = data
    
#     for doc_id, documents in dict.items():
#         return doc_id, documents


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
    while result < 10:
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
    
    end = time.time()
    spend = end - start

    print('Data Size: ' + str(len(doc_id)))
    print('Execution Time: ' + str(timedelta(seconds=spend)))
    print('--------------------------------------\n')

def showFeedbackResults(v, scores, doc_path, doc_name, weightType, relevanceType):
    first_doc = doc_name[scores.index(max(scores))]
    f = open(os.path.join(doc_path, f'{first_doc}.txt'),'r')
    ret = f.read()
    fq = v.feedback(ret)
    f.close()
    weightType = 'Feedback Queries + ' + weightType
    showResults(fq, doc_name, weightType, relevanceType)


# def showAllResults(documents, doc_name, query, weightType):
#     v = VectorSpace(documents, query, weightType)
#     scores_c = v.search('cos')
#     showResults(scores_c, doc_name, weightType, 'cos')
#     scores = v.search('eu')
#     showResults(scores, doc_name, weightType, 'eu')
#     return v, scores_c


def main():
    # doc = sys.argv[1]
    # weightType = sys.argv[2]
    # relevanceType = sys.argv[3]
    # query = sys.argv[4]

    # python3 main.py EnglishNews tf cos 'trump biden taiwan china'
    # doc_path = 'EnglishNews'
    # query = ''
    # relevanceType = 'both'
    # weightType = 'both'

    ##
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--doc', default='EnglishNews')
    parser.add_argument('-w','--weightType', default='both', help='tf or tfidf')
    parser.add_argument('-r','--relevanceType', default='both', help='cos or eu')
    parser.add_argument('-q','--query', default='Trump Biden Taiwan China', help='enter a query')
    parser.add_argument('-f', '--feedback', action="store_true")


    args = parser.parse_args()
    query = args.query.lower().split(' ')
    # doc_id, documents = doc_to_list(args.doc)
    doc_id, documents = doc_to_list('EnglishNews')


    # doc_id, documents = doc_to_dict(f'{doc}/*.txt')

    
    v = VectorSpace(documents, query, args.weightType)
    scores = v.search(args.relevanceType)

    if args.feedback:
        showFeedbackResults(v, scores, args.doc, doc_id, args.weightType, args.relevanceType)
    else:
        showResults(scores, doc_id, args.weightType, args.relevanceType)

if __name__ == '__main__':
    # main(sys.argv[1:])
    start = time.time()
    main()

    # doc_id, documents = doc_to_list('TestNews')
    
    # query = 'Trump Biden Taiwan China'
    # query = query.lower().split(' ')

    # v = VectorSpace(documents, query, weightType = 'tf')
    # scores = v.search(relevanceType = 'cos')

    # vector = v.makeVector(documents[0])
    # print(v.vectorKeywordIndex)
    # print('--------------------------')

    # for i in range(len(vector)):
    #     if vector[i] > 0:
    #         vector[i] = vector[i] / len(documents[0])
    # print(vector)
    # print('--------------------------')
    # print(len(documents[0]), len(vector))
    # print(v.documentVectors)
    # print('--------------------------')

    # for i in range(len(vector)):
    #     if vector[i] > 0 and weightType = 'tf':
    #         vector[i] = vector[i] / 268
    # print(vector)

