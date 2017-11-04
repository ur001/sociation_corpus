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
# from .utils import DictEncoder


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
        WHERE word_base_id={word_id}
        ORDER BY popularity DESC
        LIMIT {limit}
    """.format(
        word_id=word_id,
        limit=limit
    ))
    assoc_list = cursor.fetchall()
    
    return {
        # Переводим из логарифмической шкалы в линейную
        assoc_word_id: log(similarity + 1) 
        for assoc_word_id, similarity in assoc_list
    }


def normalize_vector(vector, mean_sample=5):
    """
    Возвращает нормализованный вектор деля на среднестатистическое отклонение
    """
    std = np.std(vector.values()) or 1
    return {coord: value / std for coord, value in vector.iteritems()}


def iter_corpus(words_dict, assoc_dict, min_assoc_count=5, add_positivity=True):
    """
    Итератор по корпусу через БД
    :param DictEncoder words_dict: словарь слов соответствующих номеру вектора
    :param DictEncoder assoc_dict: словарь слов компонентов вектора (ассоциаций)
    :param in min_assoc_count: минимальное число асоциаций к слову (для исключения малозаполненных слов)
    :param bool add_positivity: добавляет оценку позитивности в виде слов ":)" и  ":("
    :yileds: list[(int, float)]
    """
    words_query = Word.objects.filter(is_active=True)
    words_id2name = dict(words_query.values_list('pk', 'name'))    
    good, bad = ':)', ':('

    for word_id, name, positivity in words_query.filter(associations_count__gt=0).values_list('pk', 'name', 'positivity'):
        top_assoc_ids = get_top_assoc(word_id, limit=500)

        if len(top_assoc_ids) >= min_assoc_count:
            top_assoc_ids = normalize_vector(top_assoc_ids)  # Делит на numpy.std
            words_dict.add(words_id2name[word_id])

            assoc_vec = [
                (assoc_dict[words_id2name[word_assoc_id]], popularity)
                for word_assoc_id, popularity in top_assoc_ids.iteritems()
            ]

            # Добавляем данные о позитивности слова
            if add_positivity:
                if positivity > 0:
                    assoc_vec.append((assoc_dict[good], positivity))  

                if positivity < 0:
                    assoc_vec.append((assoc_dict[bad], -positivity))

            yield assoc_vec


def create_source_corpus(min_assoc_count=5, add_positivity=True):
    """
    Создаёт исходноый корпус и 2 словаря к нему:
    1) words_dict: к словам — векторам
    2) assoc_dict: к ассоциациям — компонентам вектора
    """
    print("Creating source corpus...")
    words_dict = DictEncoder()
    assoc_dict = DictEncoder()
    corpus = list(iter_corpus(words_dict, assoc_dict, min_assoc_count=min_assoc_count, add_positivity=add_positivity))
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
corpus, words_dict, assoc_dict = create_source_corpus(min_assoc_count=5, add_positivity=True)
save_source_corpus('sociation_org_corpus', corpus, words_dict, assoc_dict)