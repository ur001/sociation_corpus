# coding: utf-8
"""
Генерация векторной модели слов Socition2Vec 
на основе базы ассоциаций с sociation.org

ожидается дамп ассоциаций в ./source_corpus
python ./build_model.py [dest_path] (по-умолчанию ./model)
"""
import os
import sys
import numpy as np

from sociation2vec.model_builder import ModelBuilder
from sociation2vec.console_utils import query_model_print_results

SOURCE_PATH = './source_corpus'
MODEL_PATH = './model'


# Загрузка, тренировка, индексация и сохранение в папку
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args):
        out_path = args[0]
    else:
        out_path = MODEL_PATH

    model_builder = ModelBuilder(source_path=SOURCE_PATH, out_path=out_path)
    model = model_builder.get_model(
        ppmi_k=5.5, 
        svd_components=790, 
        tfidf_smooth_tf=False
    )

    print("")
    query_model_print_results(model, 'король,-мужчина,женщина')
