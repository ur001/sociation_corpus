# coding: utf-8
from operator import itemgetter
from random import randint
import numpy as np

from .utils import bow2nparray_vec, nparray2bow_vec, compare_words, save_word2vec
from .mathutils import normalize_l2



class Sociation2Vec(object):
    """
    Поиск похожих слов по факторизованной матрице PPMI из ассоциаций с Sociation.org
    """
    def __init__(
        self, 
        model_svd, corpus, 
        similarity_index, 
        words_dict, assoc_dict,
        name=None
    ):
        self.model = model_svd
        self.corpus = corpus
        self.similarity_index = similarity_index
        self.words_dict = words_dict
        self.assoc_dict = assoc_dict
        self.name = name
        
    def get_word_vector(self, word_name): 
        """
        Возвращает вектор слова
        Тут слово — это документ в терминах корпуса
        Слову соответствует вектор ассоциаций преобразованный в LSI-простраство
        :rtype: np.array
        """
        if not isinstance(word_name, str):
            return word_name
        
        sign, corpus_idx = self.get_word_idx_and_sign(word_name)
        word_vec = self.corpus[corpus_idx]
        return normalize_l2(word_vec * sign)

    def get_words_matrix(self, word_names):
        """
        Возвращает матрицу векторов для нескольких слов
        :param list[str] word_names: список слов (минус перед началом слова вычитает)
        """
        return np.vstack(list(map(self.get_word_vector, word_names)))
    
    def get_mean_words_vector(self, word_names):
        """
        Возвращает усреднённый вектор для нескольких слов
        :param list[str] word_names: список слов (минус перед началом слова вычитает)
        """
        words_marix = self.get_words_matrix(word_names)
        return normalize_l2(words_marix.mean(axis=0))
    
    def get_top_similar_to_words(self, word_names, count=15):
        """
        Возвращает топ похожих слов на указанные
        :param list[str] word_names: список слов (минус перед началом слова вычитает)
        """
        word_vec = self.get_mean_words_vector(word_names)
        return self.get_top_similar(word_vec, count, exclude=word_names)   
    
    def compare_words(
        self, word_names1, word_names2, count=10, 
        similarity_degree=0.5, separate=True, min_score=0.3  
    ):
        diff1, diff2, common = compare_words(
            dict(self.get_top_similar_to_words(word_names1, count=500)),
            dict(self.get_top_similar_to_words(word_names2, count=500)),
            count,
            exclude=set(word_names1).union(set(word_names2)),
            similarity_degree=similarity_degree,
            separate=separate,
            min_feature_value=min_score
        )
        return diff1, diff2, common

    def get_similarity_matrix(self, word_names):
        """
        Возращает матрицу близости слов (каждого с каждым)
        :param list[str] word_names: список слов (минус перед началом слова вычитает)
        """
        words_marix = self.get_words_matrix(word_names)
        return words_marix.dot(words_marix.T)

    def get_not_match_word(self, word_names):
        """Лишнее лово"""
        similarity_matrix = self.get_similarity_matrix(word_names)
        not_match_idx = np.argmin(similarity_matrix.mean(axis=0))
        return word_names[not_match_idx]
    
    def get_top_similar(self, word_vec, count=15, exclude=None):
        exclude = {self.words_dict.encode[item.strip('-')] for item in (exclude or [])}
        similarity = self.similarity_index[nparray2bow_vec(word_vec)]
        top = sorted(enumerate(similarity), key=itemgetter(1), reverse=True)[:count + len(exclude)]
        return [
            (self.words_dict.decode[word_idx], similarity) 
            for word_idx, similarity in top
            if word_idx not in exclude
        ][:count]
    
    def get_word_idx_and_sign(self, word_name):
        if word_name.startswith('-'):
            return -1, self.words_dict.encode[word_name[1:]]
        else:
            return 1, self.words_dict.encode[word_name]    
    
    def save_word2vec(self, file_name, binary=False):
        save_word2vec(file_name, self.corpus, self.words_dict.decode, binary=binary)

    def get_random_word(self):
        return self.words_dict.decode[randint(0, len(self.words_dict.decode))]        