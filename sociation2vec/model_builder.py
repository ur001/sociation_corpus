# coding: utf-8
import os
import gensim
import pickle
import numpy as np
from operator import itemgetter

from . import mathutils
from .model import Sociation2Vec
from .utils import DictEncoder


data_dir = '../data'



class ModelBuilder(object):
    def __init__(self, source_path, out_path=None):
        """
        :param str source_path: путь к исходному корпусу ассоциаций и словарям
        :param str out_path: путь для складывания модели, индекса и др. данных
        """
        self.source_path = source_path
        self.out_path = out_path
        self.words_dict = None
        self.assoc_dict = None
        self._cache = {}
        if self.out_path and not os.path.exists(self.out_path):
            os.makedirs(self.out_path) 
        
    def get_model(
        self,
        normalize_source=True, # Нормализация исходного корпуса
        ppmi=True,             # Применение PPMI-преобразования к исходному корпусу
        ppmi_cds=0.75,         # Сглаживание PPMI (1 — без сглаживания)
        ppmi_k=5,              # Сдвиг PPMI (аналог negative sampling в SGNS)
        tfidf=True,            # Применение TFIDF-преобразования поверх
        tfidf_smooth_tf=True,  # Параметр smooth_tf для TFIDF
        svd_components=800,    # Число компонент вектора слова
        normalize_svd=True,    # Нормализовать матрицу после SVD-преобразования  
    ):
        """
        Создаёт или загружает модель Sociation2vec с заданными параметрами
        PPMI cds = 0.75:
        http://www.aclweb.org/anthology/Q15-1016 
        Note that cds helps PPMI more than it does other
        methods; we suggest that this is because it reduces
        the relative impact of rare words on the distributional
        representation, thus addressing PMI’s “Achilles’ heel”. 
        
        PPMI k = 1:
        http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/
        DON’T use shifted PPMI with SVD.
        """
        model_params = []
        
        # Получаем исходный корпус ассоциаций, 
        # если нужно проводим нормализацию        
        corpus = self.get_corpus(normalize_source, model_params)
          
        # Применяем PPMI-взвешивание исходного корпуса ассоциаций (если указано)
        if ppmi:
            corpus = self.get_ppmi_corpus(corpus, ppmi_cds, ppmi_k, model_params)
            
        # Применяем TFIDF-взвешивание    
        if tfidf:
            tfidf, corpus = self.get_tfidf_corpus(corpus, tfidf_smooth_tf, model_params)
            
        # Применяем сингулярное разложение
        svd, corpus = self.get_svd(corpus, svd_components, normalize_svd, model_params)
        
        # Создаём индекс для поиска
        similarity_index = self.get_similarity_index(corpus, model_params)
        
        # Инициализируем и возвращаем модель
        return Sociation2Vec(
            svd, 
            corpus, 
            similarity_index, 
            self.words_dict, 
            self.assoc_dict,
            name=self.make_cache_key(model_params)
        )
        
    def get_corpus(self, normalize, model_params):
        cache_key = self.make_cache_key(model_params)
        if cache_key not in self._cache:
            corpus, self.words_dict, self.assoc_dict = load_source_corpus(self.source_path)
            words_count = len(self.words_dict)
            assoc_count = len(self.assoc_dict)
            self._cache[cache_key] = mathutils.corpus_to_sparse(corpus, words_count, assoc_count)
        
        corpus = self._cache[cache_key]
        
        if normalize:
            model_params.append('norm')
            cache_key = self.make_cache_key(model_params)
            if cache_key not in self._cache:
                print('Normalizing source sorpus...')
                self._cache[cache_key] = mathutils.normalize_matrix(corpus, with_mean=False)
        
        return self._cache[cache_key]
        
    def get_ppmi_corpus(self, corpus, cds, k, model_params):
        model_params.append('ppmi_{:.2f}_{:.1f}'.format(cds, k))
        cache_key = self.make_cache_key(model_params)
        
        if cache_key not in self._cache:
            print('Applying PPMI cds={:.2f}, neg={:.1f}...'.format(cds, k))
            self._cache[cache_key] = mathutils.get_ppmi_weight(corpus, cds, k)
        
        return self._cache[cache_key]
    
    def get_tfidf_corpus(self, corpus, smooth_tf, model_params):
        model_params.append('tfidf')
        if smooth_tf: model_params.append('smooth')
        cache_key = self.make_cache_key(model_params)
        
        if cache_key not in self._cache:
            print('Applying TFIDF smooth_tf={}...'.format(smooth_tf))
            self._cache[cache_key] = mathutils.get_tfidf(corpus, smooth_tf)            
        
        return self._cache[cache_key]    
        
    def get_svd(self, corpus, n_components, normalize, model_params):
        model_params.append('svd')
        if normalize: model_params.append('norm')
        model_params.append(str(n_components))
        cache_key = self.make_cache_key(model_params)
        corpus_name = 'corpus_' + cache_key
        model_name = 'model_' + cache_key
        
        try:
            # Загрузка SVD модели
            corpus = load_corpus(corpus_name, self.out_path)
            svd = load_model(model_name, self.out_path)
            
        except FileNotFoundError:
            # Создание и сохранение корпуса и SVD модели
            print('Corpus not found, creating...')            
            print('Applying SVD {}...'.format(n_components))
            
            svd, corpus = mathutils.get_svd(corpus, n_components)
            if normalize:
                corpus = mathutils.normalize_matrix(corpus)

            if self.out_path:
                save_corpus(corpus, corpus_name, self.out_path)
                save_model(svd, model_name, self.out_path)  
            
        return svd, corpus
    
    def get_similarity_index(self, corpus, model_params):
        cache_key = self.make_cache_key(model_params)
        index_name = cache_key
        # Загрузка или создание индекса
        return create_or_load_similarity_index(corpus, index_name, self.out_path)
        
    @staticmethod
    def make_cache_key(model_params):
        return '_'.join(model_params)


def save_model(model, name, data_dir=data_dir):
    file_path = os.path.join(data_dir, name + ".pkl")
    print('Saving model to {}...'.format(file_path))
    with open(file_path, 'wb') as file:  
        pickle.dump(model, file)
        

def load_model(name, data_dir=data_dir):
    file_path = os.path.join(data_dir, name + ".pkl")
    print('Loading model from {}...'.format(file_path))
    with open(file_path, 'rb') as file:  
        return pickle.load(file)
    
def save_corpus(corpus, name, data_dir=data_dir):
    file_path = os.path.join(data_dir, name + ".npy")
    print('Saving corpus to {}...'.format(file_path))
    np.save(file_path, corpus)
        

def load_corpus(name, data_dir=data_dir):
    file_path = os.path.join(data_dir, name + ".npy")
    print('Loading corpus from {}...'.format(file_path))
    return np.load(file_path)


def load_source_corpus(path):
    """
    Загружает исходный корпус со словарями из указанной директории
    """
    print("Loading source corpus...")
    # Вектора нормализованных ассоциаций слов word: {assoc1: weight1, assoc2: weight2}
    corpus = gensim.corpora.MmCorpus(os.path.join(path, 'corpus.mm'))

    print("Loading words index...")
    # Словарь слов соответствующих порядковому номеру вектора в корпусе
    words_dict = DictEncoder.load(os.path.join(path, 'words_dict.txt'))

    print("Loading assoc words index...")
    # Словарь слов соответствующих ассоциациям — компонентам вектора
    assoc_dict = DictEncoder.load(os.path.join(path, 'assoc_dict.txt'))

    print("Words: {}, Assoc words: {}".format(
        len(words_dict.decode), 
        len(assoc_dict.decode)
    ))

    return corpus, words_dict, assoc_dict


def create_similarity_index(corpus):
    if not gensim.matutils.isbow(corpus):
        corpus = mathutils.dense_to_corpus(corpus)
        
    print("Creating similarity index...")
    return gensim.similarities.MatrixSimilarity(corpus)


def save_similarity_index(similarity_index, name, data_dir=data_dir):
    file_path = os.path.join(data_dir, name + ".index")
    print('Saving similarity index to', file_path)
    similarity_index.save(file_path)


def load_similarity_index(name, data_dir=data_dir):
    file_path = os.path.join(data_dir, name + ".index")
    print('Loading similarity index from', file_path)
    return gensim.similarities.MatrixSimilarity.load(file_path)


def create_or_load_similarity_index(corpus, name, data_dir=data_dir):
    try:
        # Загрузка индекса
        similarity_index = load_similarity_index(name, data_dir=data_dir)

    except FileNotFoundError:
        # Создание индекса
        similarity_index = create_similarity_index(corpus) 
        
        if data_dir:
            save_similarity_index(similarity_index, name, data_dir=data_dir)

    return similarity_index
