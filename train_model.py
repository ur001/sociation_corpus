# coding: utf-8
"""
Загрузка исходного корпуса и создание LSI-модели с предварительным TfIdf-преобразованием
Так же создаётся индекс для поиска по LSI-корпусу
"""
import os
import sys
from gensim import corpora, models, similarities

from utils import DictEncoder, LSIAssocSimFinder


def load_source_corpus(path):
    """
    Загружает исходный корпус со словарями из указанной директории
    """
    print("Loading source corpus...")
    # Вектора нормализованных ассоциаций слов word: {assoc1: weight1, assoc2: weight2}
    corpus = corpora.MmCorpus(path + '/corpus.mm')
    print("Loading words index...")
    # Словарь слов соответствующих порядковому номеру вектора в корпусе
    words_dict = DictEncoder.load(path + '/words_dict.txt')
    print("Loading assoc words index...")

    # Словарь слов соответствующих ассоциациям — компонентам вектора
    assoc_dict = DictEncoder.load(path + '/assoc_dict.txt')
    return corpus, words_dict, assoc_dict


def train_model(corpus, words_dict, assoc_dict, num_topics=1000):
    """
    Выполняет преобразование Bow -> DfIdf -> LSI, создаёт индекс
    и создаёт объект для поиска по словам
    """
    print("Transforming TfIdf...")
    model_tfidf = models.TfidfModel(corpus)
    corpus_tfidf = model_tfidf[corpus]

    print("Creating LSI model...")
    model_lsi = models.LsiModel(corpus_tfidf, num_topics=num_topics)

    print("Transforming corpus to LSI...")
    corpus_lsi = model_lsi[corpus_tfidf]

    print("Creating similarity index for LSI corpus...")
    similarity_index = similarities.MatrixSimilarity(corpus_lsi)

    return LSIAssocSimFinder(model_lsi, model_tfidf, corpus_lsi, similarity_index, words_dict, assoc_dict)


# Загрузка, тренировка, индексация и сохранение в папку lsi_1000
if __name__ == '__main__':
    args = sys.argv[1:]

    corpus, words_dict, assoc_dict = load_source_corpus('sociation_org_corpus')
    laf = train_model(corpus, words_dict, assoc_dict, num_topics=1000)
    laf.save(args[0])