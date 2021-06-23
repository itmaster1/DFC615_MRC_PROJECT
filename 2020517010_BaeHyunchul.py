# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 09:18:35 2021

@author: 2020517010 BaeHyunchul 

20210619 - 기본 코딩 적용
       to_do :  필요한 곳에 주석 및 설명 달기 
                불용어에 대한 내용 확인 후  수정 및 추가 
"""

from newspaper import Article
from konlpy.tag import Kkma
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np

class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.twitter = Twitter()
        self.stopwords = ["라면서", "이렇다", "아이구", "과", "에서", "로", "은", "의"
                          ,"아", "휴", "아이구", "아이쿠", "아이고", "어", "나", "우리", "저희", "따라", "의해", "을", "를", "에", "의", "가",]
    
    def url2sentences(self, url):
        article = Article(url, language='ko') #언어가 한국어이므로 language='ko'로 설정
        article.download()
        article.parse()
        sentences = self.kkma.sentences(article.text)
        
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''
                
        return sentences
    
    
    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''
                
        return sentences
    
    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence is not '':
                nouns.append(' '.join([noun for noun in self.twitter.nouns(str(sentence))
                                       if noun not in self.stopwords and len(noun) > 1]))
                
        return nouns

class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []
        
    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return self.graph_sentence
    
    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}


class Rank(object):
    def get_ranks(self, graph, d=0.85): # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 # diagonal 부분을 0으로
            link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
            
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}
    
class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        
        if text[:5] in ('http:', 'https'):
            self.sentences = self.sent_tokenize.url2sentences(text)
        else:
            self.sentences = self.sent_tokenize.text2sentences(text)
        
        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
                    
        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)
        
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
        
        self.word_rank_idx =  self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
        
        
    def summarize(self, sent_num=3):
        summary = []
        index=[]
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        
        index.sort()
        for idx in index:
            summary.append(self.sentences[idx])
        
        return summary
        
    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)
        
        keywords = []
        index=[]
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)
            
        #index.sort()
        for idx in index:
            keywords.append(self.idx2word[idx])
        
        return keywords

url = 'https://www.msn.com/ko-kr/news/national/%EC%86%8D%EB%B3%B4-%EC%BF%A0%ED%8C%A1%EB%AC%BC%EB%A5%98%EC%84%BC%ED%84%B0-%ED%99%94%EC%9E%AC-%EC%8B%A4%EC%A2%85-%EC%86%8C%EB%B0%A9%EA%B4%80-%EC%88%A8%EC%A7%84-%EC%B1%84-%EB%B0%9C%EA%B2%AC/ar-AALczuy?ocid=msedgdhp&pc=U531'
textrank = TextRank(url)
for row in textrank.summarize(1): ## 1줄로 요약 
    print(row)
    print()
    
print('keywords :',textrank.keywords())  ## 10개의 keyword


# https://joolib.tistory.com/21
# https://bab2min.tistory.com/552
# https://koreascience.or.kr/article/JAKO201835146902030.pdf