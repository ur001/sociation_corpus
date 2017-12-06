# coding: utf-8
"""
Загрузка исходного корпуса и создание LSI-модели с предварительным TfIdf-преобразованием
Так же создаётся индекс для поиска по LSI-корпусу
"""
import os
import sys
from tqdm import tqdm
from gensim import corpora, models, similarities
import numpy as np
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


def train_lsi_model(corpus, words_dict, assoc_dict, num_topics=1000, power_iters=2):
    """
    Выполняет преобразование Bow -> DfIdf -> LSI, создаёт индекс
    и создаёт объект для поиска по словам
    """
    print("Transforming TfIdf...")
    model_tfidf = models.TfidfModel(corpus)
    corpus_tfidf = model_tfidf[corpus]

    print("Creating LSI model (num_topics={}, power_iters={})...".format(num_topics, power_iters))
    chunksize = len(corpus_tfidf)
    if power_iters:
        model_lsi = models.LsiModel(corpus_tfidf, num_topics=num_topics, power_iters=power_iters, onepass=False, extra_samples=500)
    else:
        model_lsi = models.LsiModel(corpus_tfidf, num_topics=num_topics, chunksize=chunksize)

    print("Transforming corpus to LSI...")
    corpus_lsi = model_lsi[corpus_tfidf]

    print("Creating similarity index for LSI corpus...")
    similarity_index = similarities.MatrixSimilarity(corpus_lsi)

    return LSIAssocSimFinder(model_lsi, model_tfidf, corpus_lsi, similarity_index, words_dict, assoc_dict)


# Загрузка, тренировка, индексация и сохранение в папку lsi_1000
if __name__ == '__main__':
    args = sys.argv[1:]

    mode = 'lsi'
    if len(args) > 1:
        mode = args[1]

    corpus, words_dict, assoc_dict = load_source_corpus('sociation_org_corpus')
    if mode == 'lsi':
        num_topics = int(args[2]) if len(args) > 2 else 800
        power_iters = int(args[3]) if len(args) > 3 else None
        model = train_lsi_model(corpus, words_dict, assoc_dict, num_topics=num_topics, power_iters=power_iters)

    model.save(args[0])