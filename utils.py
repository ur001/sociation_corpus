# coding: utf-8
"""
Создание исходного корпуса Sociation.org
Выполняется в консоли. Код для ознакомления
"""
import os
from collections import Counter
from operator import itemgetter
from random import randint
import numpy as np
from gensim import corpora, models, similarities


class DictEncoder(object):
    """
    Вспомогательная фигня для создания/хранения словарей
    Каждому новому слову сопоставляет целое число

    >>> my_dict = DictEncoder()
    >>> print my_dict[u'слово']
    0
    >>> print my_dict[u'другое слово']
    1
    >>> print my_dict.decode[0]
    слово
    >>> my_dict.save(file_name)
    """
    def __init__(self):
        self.encode = {}
        self.decode = []

    def add(self, item):
        if item in self.encode:
            return self.encode[item]

        index = len(self.decode)
        self.decode.append(item)
        self.encode[item] = index
        return index

    def save(self, file_name):
        with open(file_name, 'w') as file:
            for item in self.decode:
                file.write(item + '\n')

    @classmethod
    def load(cls, file_name):
        encoder = cls()
        with open(file_name, 'r') as file:
            for item in file:
                encoder.add(item.strip())
        return encoder

    def __getitem__(self, item):
        return self.add(item)

    def __contains__(self, item):
        return item in self.encode


class LSIAssocSimFinder(object):
    """
    Поиск похожих слов по LSI-модели Sociation.org
    """
    def __init__(self, model_lsi, model_tfidf, corpus_lsi, similarity_index, words_dict, assoc_dict):
        self.model_lsi = model_lsi
        self.model_tfidf = model_tfidf
        self.corpus_lsi = corpus_lsi
        self.similarity_index = similarity_index
        self.words_dict = words_dict
        self.assoc_dict = assoc_dict

    def save(self, name):
        if not os.path.exists(name):
            os.makedirs(name)

        print ("Saving LSI model...")
        self.model_lsi.save(name + '/model.lsi')
        print ("Saving TfIdf model...")
        self.model_tfidf.save(name + '/model.tfidf')
        print ("Saving LSI corpus...")
        corpora.BleiCorpus.serialize(name + '/corpus_lsi.lda-c', self.corpus_lsi)
        print ("Saving similarity index...")
        self.similarity_index.save(name + '/corpus.index')
        print ("Saving words dict...")
        self.words_dict.save(name + '/words_dict.txt')     
        print ("Saving assoc dict...")
        self.assoc_dict.save(name + '/assoc_dict.txt')            
        
    @classmethod
    def load(cls, name):
        print ("Loading LSI model...")
        model_lsi = models.LsiModel.load(name + '/model.lsi')
        print ("Loading TfIdf model...")
        model_tfidf = models.TfidfModel.load(name + '/model.tfidf')
        print ("Loading LSI corpus...")
        corpus_lsi = corpora.BleiCorpus(name + '/corpus_lsi.lda-c')
        print ("Loading similarity index...")
        similarity_index = similarities.MatrixSimilarity.load(name + '/corpus.index')
        print ("Loading words index...")
        words_dict = DictEncoder.load(name + '/words_dict.txt')
        print ("Loading assoc words index...")
        assoc_dict = DictEncoder.load(name + '/assoc_dict.txt')        
        return cls(model_lsi, model_tfidf, corpus_lsi, similarity_index, words_dict, assoc_dict)

    def get_word_lsi_vector(self, word_name): 
        """
        Возвращает LSI-вектор проиндексированного слова
        Тут слово — это документ в терминах корпуса
        Слову соответствует вектор ассоциаций преобразованный в LSI-простраство
        :rtype: np.array
        """
        mul = -1 if word_name.startswith('-') else 1
        corpus_idx = self.words_dict.encode[word_name.strip('-')]
        word_vec = np.array(list(map(itemgetter(1), self.corpus_lsi[corpus_idx])))
        return word_vec * mul

    def get_lsi_assoc_vector(self, word_names, tfidf=False):
        """
        Возвращает LSI-вектор из списка слов буд-то бы являющихся ассоциациями
        :rtype: list[tuple[int, float]]
        """
        bow = (self.assoc_dict[name] for name in word_names if name in self.assoc_dict)
        vec_count = Counter(bow).items()
        if tfidf:
            vec_tfidf = self.model_tfidf[vec_count]
            return self.model_lsi[vec_tfidf]
        else:
            return self.model_lsi[vec_count]

    def get_top_similar_to_words(self, word_names, count=15):
        """
        Выводит топ похожих слов на указанные
        Можно несколько через щапятую и через минус: король,-мужчина,женщина = королева
        """
        word_vec = sum(map(self.get_word_lsi_vector, word_names))
        word_vec = [(idx, val) for idx, val in enumerate(word_vec)]
        return self.get_top_similar(word_vec, count, exclude=word_names)

    def get_top_similar_for_assoc(self, word_names, count=15):
        """
        Выводит топ слов интерпетируя входные слова как ассоциации: мята,лайм = мохито
        """
        word_vec = self.get_lsi_assoc_vector(word_names)
        return self.get_top_similar(word_vec, count, exclude=word_names)

    def get_top_similar(self, word_vec, count=15, exclude=None):
        exclude = set(self.words_dict.encode[item.strip('-')] for item in (exclude or []))
        similarity = self.similarity_index[word_vec]
        top = sorted(enumerate(similarity), key=itemgetter(1), reverse=True)[:count + len(exclude)]
        return (
            (self.words_dict.decode[word_idx], similarity) 
            for word_idx, similarity in top
            if word_idx not in exclude
        )

    def get_random_word(self):
        return self.words_dict.decode[randint(0, len(self.words_dict.decode))]
