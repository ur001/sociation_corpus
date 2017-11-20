# coding: utf-8
"""
Загрузка корпуса близости слов и создание индекса для поиска
"""
import os
import sys
from tqdm import tqdm
from gensim import corpora, models, similarities
import numpy as np
from utils import DictEncoder, SimSimFinder


def main(source_path, dest_path):
    corpus, words_dict = load_similarity_corpus(source_path)
    ssf = make_index_and_init_ssf(corpus, words_dict)
    ssf.save(dest_path)


def load_similarity_corpus(path):
    """
    Загружает исходный корпус со словарями из указанной директории
    """
    print("Loading source corpus...")
    # Вектора схожести слов слов word: {assoc1: weight1, assoc2: weight2}
    corpus = corpora.MmCorpus(path + '/corpus.mm')

    print("Loading words index...")
    # Словарь слов соответствующих порядковому номеру вектора в корпусе
    words_dict = DictEncoder.load(path + '/words_dict.txt')

    return corpus, words_dict


def make_index_and_init_ssf(corpus, words_dict):
    """
    Индексирует корпус созданный по индексу и возвращает инициализиованную модель SimSimFinder
    """
    # print("Creating LSI model...")
    # model_lsi = models.LsiModel(corpus, num_topics=1000, dtype=np.float32)

    # print("Transforming corpus to LSI...")
    # corpus = model_lsi[corpus]

    print("Creating similarity index for similarity corpus...")
    words_count = len(words_dict.decode)
    similarity_index = similarities.SparseMatrixSimilarity(corpus, num_features=words_count, num_terms=words_count)
    return SimSimFinder(corpus, similarity_index, words_dict)



# Загрузка,индексация и сохранение в папку
if __name__ == '__main__':
    args = sys.argv[1:]
    main('sociation_similarity_corpus', 'sociation_similarity_index')

