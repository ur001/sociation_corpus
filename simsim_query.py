# coding: utf-8
"""
Поиск похожих слов
"""
import sys
from random import choice

from utils import DictEncoder, SimSimFinder


def main(path, command=None, count=15, is_interactive=False):
    finder = SimSimFinder.load(path)
    if command:
        do_command(finder, command, count)    

    if interactive:
        interactive(finder, count)


def do_command(finder, command, count=15):
    if '|' in command: # Сравнение слов
        word_names1, word_names2 = command.split('|')
        diff1, diff2, common = finder.compare_words(
            dict(finder.get_top_similar_to_words(word_names1.split(','), 500)),
            dict(finder.get_top_similar_to_words(word_names2.split(','), 500)),
            count,
            exclude=set(word_names1.split(',')).union(set(word_names2.split(',')))
        )
        diff1, diff2, common 
        print_results(word_names1, diff1)
        print_results(word_names2, diff2)
        print_results(command.replace('|', ' + '), common)

    else:  # поиск похожих. Слово или слова через запятую (столица,украина; король,-мужчина, женщина)
        results = finder.get_top_similar_to_words(command.split(','), count)
        print_results(command, results)


def print_results(title, results):
    print(title)
    print ("=" * 20)
    
    for word_name, similarity in results:    
        print("{:0.3f}\t{}".format(similarity, word_name))

    print("")
          

def interactive(finder, count=15):
    while True:
        print('\nВведите слово или нажмите [Enter] для показа случайного.\nДля выхода введите [q] или [x].')
        command = input().lower() or get_random_command(finder)

        if command in {'q', 'x'}:
            return

        try:
            do_command(finder, command, count)
        except KeyError:
            print ("Слово не найдено\n")
            

def get_random_command(finder):
    return ",".join(
        (choice(['', '', '', '', '-']) if word_num >= 1 else '') + finder.get_random_word()
        for word_num in range(choice([1,1,1,1,1,2,2,2,3]))
    )


if __name__ == '__main__':
    args = sys.argv[1:]
    path, command, count, is_interactive = 'sociation_similarity_index', None, 15, False

    if len(args) >= 1:
        path = args[0]

    if len(args) >= 2 and args[1] != 'interactive':
        command = args[1].lower()

    if len(args) >= 3:
        try:
            count = int(args[2])
        except:
            pass

    if 'interactive' in args:
        is_interactive = True

    main(path, command, count, is_interactive)