# coding: utf-8
import sys
import os
from tqdm import tqdm
from gensim import corpora, models, similarities
import numpy as np
from utils import DictEncoder, LSIAssocSimFinder


def iter_corpus(laf, filter_vec):
    """
    Итератор по близким словам полученным с помощью LSIAssocSimFinder
    :param LSIAssocSimFinder laf: 
    :param (int, float) -> bool filter_vec: функция для фильтрации компонентов вектора слова
    :yileds: list[(int, float)]
    """
    for word_id, word_name in enumerate(laf.words_dict.decode):
        try:
            yield [
                (laf.words_dict.encode[similar_word], similarity)
                for similar_word, similarity 
                in filter(filter_vec, laf.get_top_similar_to_words([word_name], 200))
            ]
        except ValueError:
            print(word_id, word_name)


def create_similarity_corpus(path,  min_similarity=0.1):
    def filter_best(word):
        word, similarity = word
        return similarity > min_similarity

    laf = LSIAssocSimFinder.load(path)

    print("Creating similarity corpus...")    
    corpus = list(tqdm(iter_corpus(laf, filter_best), total=len(laf.words_dict.decode)))
    return corpus, laf.words_dict


def save_similarity_corpus(path, corpus, words_dict):
    if not os.path.exists(path):
        os.makedirs(path)    

    print("Saving similarity corpus...")
    corpora.MmCorpus.serialize(path + '/corpus.mm', corpus)
    print("Saving words dict...")
    words_dict.save(path + '/words_dict.txt')


def main(source_path, dest_path, min_similarity=0.05):
    corpus, words_dict = create_similarity_corpus(source_path, min_similarity)
    save_similarity_corpus(dest_path, corpus, words_dict)


if __name__ == '__main__':
    args = sys.argv[1:]
    source_path = args[0]

    dest_path='sociation_similarity_corpus'
    if len(args) > 1:
        dest_path = args[1]

    main(source_path, dest_path)