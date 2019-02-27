# coding: utf-8
import gensim
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import scale
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_array


random_state = 777


def get_csr_matrix_from_xyval_list(xyval_list, dtype=np.float32):
    """Создаёт CSR матрицу из списка [(x, y, value), ...]"""
    x, y, value = zip(*xyval_list)
    matrix = csr_matrix((value, (x, y)), dtype=dtype)
    matrix.eliminate_zeros()
    return matrix


def multiply_by_rows(matrix, row_coefs):
    """Умножает разреженную матрицу построчно на вектор"""
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)


def multiply_by_columns(matrix, col_coefs):
    """Умножает разреженную матрицу поколоночно на вектор"""
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())


def get_ppmi_weight(matrix, cds=0.75, k=5):
    """Рассчёт PPMI для разреженной матрицы"""
    marginal_rows = np.array(matrix.sum(axis=1))[:, 0]
    marginal_cols = np.array(matrix.sum(axis=0))[0, :] ** cds
    pmi = matrix * marginal_cols.sum()
    pmi = multiply_by_rows(pmi, np.reciprocal(marginal_rows))
    pmi = multiply_by_columns(pmi, np.reciprocal(marginal_cols))
    pmi.data = (np.log2(pmi.data) - np.log(k)).clip(0.0)
    return pmi


def get_svd(matrix, n_components, n_iter=10):
    """Факторизация матрицы c помощью SVD"""
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=random_state)
    return svd, svd.fit_transform(matrix)

# def get_svd(matrix, n_components, eigenvalue_weight=0.5, n_iter=10):
#     check_array(matrix, accept_sparse=['csr', 'csc'])
#     svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=random_state)
#     U, Sigma, VT = randomized_svd(matrix, n_components, n_iter=n_iter, random_state=random_state)
#     return svd, U * (Sigma ** eigenvalue_weight)


def get_tfidf(matrix, smooth_idf=True):
    """
    Применение TfIdf к матрице, 
    TODO: попробовать оставить один IDF
    """
    tfidf = TfidfTransformer(smooth_idf=smooth_idf)
    return tfidf, tfidf.fit_transform(matrix)


def normalize_matrix(matrix, with_mean=True):
    """
    Нормализация матрицы (aka «стандартизация», -mean/std)
    Дважды транспонируется так как scale не хочет нормализовать 
    разреженные матрицы по строкам
    """
    return scale(matrix.T, with_mean=with_mean).T


def normalize_l2(vec):
    norm = np.linalg.norm(vec)
    if norm != 0: 
       vec = vec / norm
    return vec    


def corpus_to_sparse(corpus, words_count, assoc_count):
    return gensim.matutils.corpus2csc(corpus, num_terms=assoc_count, num_docs=words_count).T


def sparse_to_corpus(matrix):
    return gensim.matutils.Sparse2Corpus(matrix.T)


def dense_to_corpus(matrix):
    return gensim.matutils.Dense2Corpus(matrix.T)    