# coding: utf-8
import pandas as pd
from operator import itemgetter


def get_associations(word_name, corpus, words_dict, assoc_dict, count=10):
    """Ассоциации к слову"""
    associations = [
        (assoc_dict.decode[assoc_id], popularity)
        for assoc_id, popularity
        in sorted(corpus[words_dict.encode[word_name]], key=itemgetter(1), reverse=True)
    ]
    count = count or len(associations)
    return pd.DataFrame(associations[:count], columns=['word', 'popularity']).set_index('word')


def sim2df(similar):
    return pd.DataFrame(similar, columns=['word', 'similarity']).set_index('word').style.bar()


def build_get_similar(sociation2vec):
    def get_similar(word_names, count=10):
        """Ассоциации к слову"""
        return sim2df(sociation2vec.get_top_similar_to_words(word_names.lower().split(','), count=count))
    return get_similar


def build_doesnt_match(sociation2vec):
    def print_not_match_word(word_names):
        """Лишнее лово"""
        print(sociation2vec.get_not_match_word(word_names.lower().split(',')))
    return print_not_match_word    


def build_compare(sociation2vec):
    def compare(
        word_names1, word_names2, count=10,
        similarity_degree=0.5, separate=True, min_score=0.3         
    ):
        """Сравнение слов: обшие и разница"""
        diff1, diff2, common = sociation2vec.compare_words(
            word_names1.lower().split(','), 
            word_names2.lower().split(','), 
            count,
            similarity_degree=similarity_degree,
            separate=separate,
            min_score=min_score
        )
        
        blank = [('', '')]
        diff1 = (diff1 + blank * count)[:count]
        diff2 = (diff2 + blank * count)[:count]
        common = (common + blank * count)[:count]
        
        return pd.DataFrame([
            [word1, score1, word_common, score_common, word2, score2]
            for (word1, score1), (word_common, score_common), (word2, score2)
            in zip(diff1, common, diff2)
        ], columns=[
            word_names1, 'score1',
            'общие', 'score_common',
            word_names2, 'score2',
        ])
    return compare    