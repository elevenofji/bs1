import jieba
import os
import codecs
import re
from pprint import pprint
from gensim import corpora,models,similarities

from collections import defaultdict
#import sklearn
#import jieba.analyse


def word_segment(filepath,segpath):
    files = os.listdir(filepath)
    for txt in files:
        filename = os.path.join('%s/%s'%(filepath,txt))
        seg_file = os.path.join('%s/%s'%(segpath,txt))
        f = codecs.open(filename, 'r','utf-8')
        fw = codecs.open(seg_file, 'w', 'utf-8')
        # 去掉所有换行符号
        s = ''
        for line in f:
            s += str(line)
        seg_list = jieba.lcut(s)
        fw.write("/".join(seg_list))
        f.close()
        fw.close()

def seg_to_one(filepath):
    files = os.listdir(filepath)
    documents = []
    for txt in files:
        filename = os.path.join('%s/%s' % (filepath, txt))
        f = codecs.open(filename,'r','utf-8')
        for line in f:
            documents.append(line)
    return documents

def save_bow_dic(segpath):
    documents = seg_to_one(segpath)
    texts = [[word for word in document.split('/')] for document in documents]
    fre = defaultdict(int)
    for text in texts:
        for word in text:
            fre[word] += 1
    # 过滤频次比较低的
    texts = [[word for word in text if fre[word] > 10] for text in texts]
    dictionary = corpora.Dictionary(texts)
    dictionary.save("dict_corpus.txt")
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('corpus.mm',corpus)
'''
    with open('./corpus.txt','w') as f:
        for line in corpus:
            for tu in line:
                f.write(str(tu[0]) + ',' + str(tu[1]))
                f.write(' ')
            f.write('\n')
'''

def read_corpus(filepath):#没用
    corpus = []
    with open(filepath, 'r') as f:
        for line in f:
            line1 = line.strip().split(' ')
            line2 = []
            for string in line1:
                string1 = string.split(',')
                tu = (int(string1[0]),int(string1[1]))
                line2.append(tu)
            corpus.append(line2)
    return corpus

def save_tfidf(corpus):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    for item in corpus_tfidf:
        T1 = item
        print(item)
    tfidf.save("data.tfidf")


if __name__ == '__main__':
    filepath = './book'
    segpath = './seg'
    #word_segment(filepath, segpath)
    save_bow_dic(segpath)
    dictionary = corpora.Dictionary.load('corpus.dic')
    corpus = corpora.MmCorpus('corpus.mm')
    #save_tfidf(corpus)
    tfidf_model = models.TfidfModel.load("data.tfidf")
    corpus_tfidf = tfidf_model[corpus]
    s = list(corpus_tfidf)
    pprint(list(corpus_tfidf))
    #pprint(list(corpus))

'''

    lda_model = models.LdaModel(corpus_tfidf,id2word=dictionary,num_topics = 4)
    corpus_lda = lda_model[corpus_tfidf]
    nodes = list(corpus_lda)
    print(nodes)

'''
