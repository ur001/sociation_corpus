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

    def __len__(self):
        return len(self.decode)


class WordsComparator(object):
    def __init__(self, feature1_max, feature2_max, similarity_degree=1/3):
        self.feature1_max = feature1_max
        self.feature2_max = feature2_max
        self.prod_max = feature1_max * feature2_max
        self.similarity_degree = similarity_degree

    def __call__(self, feature1, feature2):
        diff1_value = feature1 * (self.feature2_max - feature2) / self.prod_max
        diff2_value = feature2 * (self.feature1_max - feature1) / self.prod_max
        common_value = (feature1 * feature2 / self.prod_max) ** self.similarity_degree
        return diff1_value, diff2_value, common_value


def bow2nparray_vec(vec):
    return np.array(list(map(itemgetter(1), vec)))


def nparray2bow_vec(vec):
    return [(idx, val) for idx, val in enumerate(vec) if val]



def compare_words(
    word1_features, 
    word2_features, 
    count=10, 
    exclude=set(), 
    similarity_degree=0.5, 
    separate=False,
    min_feature_value=0.3        
):
    """
    Сравнение двух слов на основе списка похожих (или вообще каких-либо фич слова).
    Возвращает 3 списка: характерные для первого слова, второго и общие
    :param dict[int, float] word1_features: фичи первого слова: словарь {feature: value}
    :param dict[int, float] word2_features: фичи второго слова: словарь {feature: value}
    :param in count: число слов в результах
    :param float similarity_degree: число 0..1. 1 — полное разделение слов, 0 — максимальный поиск сходства
    :param bool separate: «срогое разделение» — запрет попадания одного слова в несколько колонок
    :param float min_feature_value: минимальное значение 
    """
    diff1, diff2, common = {}, {}, {}  # Характерное для первого слова, для второго и общее
    features = set(word1_features.keys()).union(word2_features.keys())        

    for feature in features:
        if feature in exclude:
            continue

        feature1 = word1_features.get(feature, 0)
        feature2 = word2_features.get(feature, 0)
        if feature1 < min_feature_value and feature2 < min_feature_value:
            continue

        diff1_value = feature1 * (1 - feature2)
        diff2_value = feature2 * (1 - feature1)
        common_value = (feature1 * feature2) ** similarity_degree
        max_value = max(diff1_value, diff2_value, common_value)

        if diff1_value == max_value or not separate:
            diff1[feature] = diff1_value

        if diff2_value == max_value or not separate:
            diff2[feature] = diff2_value

        if common_value == max_value or not separate:
            common[feature] = common_value                

    return (
        sorted(diff1.items(), key=itemgetter(1), reverse=True)[:count],
        sorted(diff2.items(), key=itemgetter(1), reverse=True)[:count],
        sorted(common.items(), key=itemgetter(1), reverse=True)[:count],
    ) 


def nparray2str(value, binary=False):
    if binary:
        return value.tostring()
    else:
        return ' '.join(map(str, value))


def save_word2vec(file_name, corpus, dictionary, binary=False):
    with open(file_name, 'w') as file:
        file.write('{} {}\n'.format(len(dictionary), len(corpus[0])))
        for word_idx, vector in enumerate(corpus):
            word = dictionary[word_idx].replace(' ', '_')    
            file.write('{} {}\n'.format(word, nparray2str(vector, binary)))


def read_word2vec(path):
    words_dict = DictEncoder()
    corpus = []
    with open(path, 'r') as file:
        for line in file:
            part1, part2 = line.strip().split(' ', 1)
            if not part1.isdigit():
                word_name, word_vector = part1, np.fromstring(part2, dtype="float32", sep=" ")
                words_dict.add(word_name)
                corpus.append(word_vector)
                
    return np.vstack(corpus), words_dict            