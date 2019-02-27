# coding: utf-8
"""
Создание исходного корпуса Sociation.org
Выполняется в консоли. Код для ознакомления
Python2! Остальные скрипты на python3
"""
import os
from subprocess import call
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


words_query = Word.objects.filter(is_active=True)


def iter_corpus(words_dict, assoc_dict, min_assoc_count=5, add_positivity=True):
    """
    Итератор по корпусу через БД
    :param DictEncoder words_dict: словарь слов соответствующих номеру вектора
    :param DictEncoder assoc_dict: словарь слов компонентов вектора (ассоциаций)
    :param in min_assoc_count: минимальное число асоциаций к слову (для исключения малозаполненных слов)
    :param bool add_positivity: добавляет оценку позитивности в виде слов ":)" и  ":("
    :yileds: list[(int, float)]
    """
    
    words_id2name = dict(words_query.values_list('pk', 'name'))    
    good, bad = ':)', ':('

    if add_positivity:
        words_id2name[good] = good
        words_id2name[bad] = bad

    for word_id, name, positivity in words_query.filter(associations_count__gt=0).values_list('pk', 'name', 'positivity'):
        top_assoc = get_top_assoc(word_id, limit=500)

        if len(top_assoc) >= min_assoc_count:
            # Добавляем данные о позитивности слова
            if add_positivity:
                if positivity > 0:
                    top_assoc[good] = positivity

                if positivity < 0:
                    top_assoc[bad] = -positivity

            words_dict.add(words_id2name[word_id])

            assoc_vec = [
                (assoc_dict[words_id2name[word_assoc_id]], popularity)
                for word_assoc_id, popularity in top_assoc.iteritems()
            ]

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
    params = dict(min_assoc_count=min_assoc_count, add_positivity=add_positivity)
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
corpus, words_dict, assoc_dict = create_source_corpus(min_assoc_count=4, add_positivity=True)
save_source_corpus('source_corpus', corpus, words_dict, assoc_dict)
call(["zip", "-r", "source_corpus.zip", "source_corpus"])
call(["rm", "-rf", "source_corpus"])
# zip -r source_corpus.zip source_corpus
# unzip source_corpus.zip
# sociation-copy-source
# sociation-build-vec