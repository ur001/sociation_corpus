# coding: utf-8
"""
Создание исходного корпуса Sociation.org
Выполняется в консоли. Код для ознакомления
"""
import os
import numpy as np
from math import sqrt, pow, log, ceil
from django.db import connection
from gensim import corpora

from __future__ import print_function


class DictEncoder(object):
    """
    Вспомогательная фигня для создания/хранения словарей
    Каждому новому слову сопоставляет целое число
    Дублирует .utils.DictEncoder для python2

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
        with open(file_name, 'wb') as file:
            for item in self.decode:
                file.write(item.encode('utf-8') + '\n')

    @classmethod
    def load(cls, file_name):
        encoder = cls()
        with open(file_name, 'r') as file:
            for item in file:
                encoder.add(item.decode("utf-8").strip())
        return encoder

    def __getitem__(self, item):
        return self.add(item)

    def __contains__(self, item):
        return item in self.encode


def get_top_assoc(word_id, limit=50):
    """
    Возвращает вектор ближайших `limit` ассоциаций к слову 
    в виде словаря {id_слова: ln(популярность)}. 
    К популрности ассоциации применяется логарифмирование.
    """
    cursor = connection.cursor()
    cursor.execute("""
        SELECT word_assoc_id, popularity
        FROM sym_assoc 
        LEFT JOIN words_word w ON w.id = word_assoc_id
        WHERE word_base_id={word_id} AND popularity > 0
        ORDER BY popularity DESC
        LIMIT {limit}
    """.format(
        word_id=word_id,
        limit=limit
    ))

    return dict(cursor.fetchall())


def log1p_vector(vector):
    return {
        # Переводим из логарифмической шкалы в линейную
        coord: log(value + 1) 
        for coord, value in vector.iteritems()
    }    


def normalize_vector(vector):
    """
    Возвращает нормализованный вектор деля на среднестатистическое отклонение
    """
    std = np.std(vector.values() + [0])
    return {coord: value / std for coord, value in vector.iteritems()}


CDS = 0.75  # коэффициент для PPMI
words_query = Word.objects.filter(is_active=True)
words_popularity = dict(words_query.values_list('pk', 'sort_index'))
pow_cds = lambda iterable: (abs(value) ** CDS for value in iterable)
words_popularity_sum = sum(pow_cds(words_popularity.itervalues()))


def ppmize_word_assoc_vector(word_id, vector, cds=CDS):
    """
    Взвешивает вектор ассоциаций к слову word_id с помощью PPMI
    """
    word_popularity = words_popularity[word_id]
    result = {}
    for word_assoc_id, assoc_popularity in vector.iteritems():
        word_assoc_popularity = words_popularity[word_assoc_id] ** cds
        pmi = np.log2(assoc_popularity * words_popularity_sum / word_popularity / word_assoc_popularity)
        if pmi > 0:
            result[word_assoc_id] = pmi
    return result


def iter_corpus(words_dict, assoc_dict, min_assoc_count=5, add_positivity=True, normalize=None):
    """
    Итератор по корпусу через БД
    :param DictEncoder words_dict: словарь слов соответствующих номеру вектора
    :param DictEncoder assoc_dict: словарь слов компонентов вектора (ассоциаций)
    :param in min_assoc_count: минимальное число асоциаций к слову (для исключения малозаполненных слов)
    :param bool add_positivity: добавляет оценку позитивности в виде слов ":)" и  ":("
    :param str | None normalize: нормализация: None, 'log_norm', 'ppmi'
    :yileds: list[(int, float)]
    """
    global words_popularity_sum
    
    words_id2name = dict(words_query.values_list('pk', 'name'))    
    good, bad = ':)', ':('

    if add_positivity:
        words_id2name[good] = good
        words_id2name[bad] = bad
        words_popularity[good] = sum(pow_cds(words_query.filter(positivity__gt=0).values_list('positivity', flat=True)))
        words_popularity[bad] = sum(pow_cds(words_query.filter(positivity__lt=0).values_list('positivity', flat=True)))
        words_popularity_sum += words_popularity[good] + words_popularity[bad]

    for word_id, name, positivity in words_query.filter(associations_count__gt=0).values_list('pk', 'name', 'positivity'):
        top_assoc = get_top_assoc(word_id, limit=500)

        if normalize is None:
            normalize_assoc_vector = lambda word_id, vector: vector
        elif normalize == 'log_norm':
            normalize_assoc_vector = lambda word_id, vector: normalize_vector(log1p_vector(vector)) 
        elif normalize == 'ppmi':
            normalize_assoc_vector = lambda word_id, vector: ppmize_word_assoc_vector(word_id, vector)
        else:
            raise Exception('unknown normalize "{}"'.format(normalize))

        if len(top_assoc) >= min_assoc_count:
            # Добавляем данные о позитивности слова
            if add_positivity:
                if positivity > 0:
                    top_assoc[good] = positivity

                if positivity < 0:
                    top_assoc[bad] = -positivity

            top_assoc = normalize_assoc_vector(word_id, top_assoc) 
            words_dict.add(words_id2name[word_id])

            assoc_vec = [
                (assoc_dict[words_id2name[word_assoc_id]], popularity)
                for word_assoc_id, popularity in top_assoc.iteritems()
            ]

            yield assoc_vec


def create_source_corpus(min_assoc_count=5, add_positivity=True, normalize=None):
    """
    Создаёт исходноый корпус и 2 словаря к нему:
    1) words_dict: к словам — векторам
    2) assoc_dict: к ассоциациям — компонентам вектора
    """
    print("Creating source corpus...")
    words_dict = DictEncoder()
    assoc_dict = DictEncoder()
    params = dict(min_assoc_count=min_assoc_count, add_positivity=add_positivity, normalize=normalize)
    corpus = list(iter_corpus(words_dict, assoc_dict, **params))
    return corpus, words_dict, assoc_dict


def save_source_corpus(path, corpus, words_dict, assoc_dict):
    """
    Сохараняет исходный корпус со словарями в указанную директорию
    """
    if not os.path.exists(path):
        os.makedirs(path)    
    print("Saving source corpus...")
    corpora.MmCorpus.serialize(path + '/corpus.mm', corpus)
    print("Saving words dict...")
    words_dict.save(path + '/words_dict.txt')
    print("Saving assoc dict...")
    assoc_dict.save(path + '/assoc_dict.txt')
 

# Создание и сохранение исходного корпуса со словарями и сохранение в папку ./sociation_org_corpus
corpus, words_dict, assoc_dict = create_source_corpus(min_assoc_count=5, add_positivity=True, normalize='ppmi')
save_source_corpus('sociation_org_corpus', corpus, words_dict, assoc_dict)
# zip -r sociation_org_corpus.zip sociation_org_corpus