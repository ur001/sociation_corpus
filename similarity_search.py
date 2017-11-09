# coding: utf-8
"""
Поиск похожих слов
"""
import sys
from random import choice

from utils import DictEncoder, LSIAssocSimFinder


def main(path, word_names=None, count=15, is_interactive=False, assoc_mode=False):
    laf = LSIAssocSimFinder.load(path)
    if word_names:
        print_top_similar(laf, word_names, count, assoc_mode=assoc_mode)    

    if interactive:
        interactive(laf, count, assoc_mode=assoc_mode)


def print_top_similar(laf, word_names=None, count=15, assoc_mode=False):
    print(word_names)
    print ("=" * 20)
    get_top_similar = laf.get_top_similar_for_assoc if assoc_mode else laf.get_top_similar_to_words

    if word_names:
        # Слово или слова через запятую (столица,украина; король,-мужчина, женщина)
        word_names = word_names.split(',')
    else:
        # Случайное слово
        words_count = laf.get_random_word()
        word_names = []

    for word_name, similarity in get_top_similar(word_names, count):    
        print("{:0.3f}\t{}".format(similarity, word_name))

    print("")


def interactive(laf, count=15, assoc_mode=False):
    # Метод поиска: по словам или по ассоциациям
    print('\nВведите слово или нажмите [Enter] для показа случайного.\nДля выхода введите [q] или [x].')
    word_names = input().lower()
    while True:
        if word_names in {'q', 'x'}:
            return

        word_names = word_names or get_random_word_names(laf)
        try:
            print_top_similar(laf, word_names)
        except KeyError:
            print ("Слово не найдено\n")
            
        word_names = input().lower() 


def get_random_word_names(laf):
    return ",".join(
        (choice(['', '', '', '', '-']) if word_num >= 1 else '') + laf.get_random_word()
        for word_num in range(choice([1,1,1,1,1,2,2,2,3]))
    )


if __name__ == '__main__':
    args = sys.argv[1:]
    path, word_names, count, is_interactive = 'lsi_1000', None, 15, False

    if len(args) >= 1:
        path = args[0]

    if len(args) >= 2 and args[1] != 'interactive':
        word_names = args[1].lower()

    if len(args) >= 3:
        try:
            count = int(args[2])
        except:
            pass

    if 'interactive' in args:
        is_interactive = True

    main(path, word_names, count, is_interactive, assoc_mode=False)