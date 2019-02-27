# coding: utf-8
"""
Поиск похожих слов
"""
import sys
from random import choice

from sociation2vec.model_builder import ModelBuilder
from sociation2vec.console_utils import query_model_print_results


SOURCE_PATH = './source_corpus'
MODEL_PATH = './model'


def main(source_path, out_path, command=None, count=15, is_interactive=False):
    model_builder = ModelBuilder(source_path=source_path, out_path=out_path)
    model = model_builder.get_model(ppmi_k=5.5, svd_components=790, tfidf_smooth_tf=False)
    
    if command:
        do_command(model, command, count)    

    if interactive:
        interactive(model, count)


def do_command(model, command, count=15):
    if command in {'p', '?'}:
        print ("=" * 20)
        print(sys.argv[1])
        print ("=" * 20)
        
    elif '|' in command: # Сравнение слов
        word_names1, word_names2 = command.split('|')
        diff1, diff2, common = model.compare_words(
            word_names1.lower().split(','), 
            word_names2.lower().split(','), 
            count,
            similarity_degree=0.5, separate=True, min_score=0.3 
        )
        print_results(word_names1, diff1)
        print_results(word_names2, diff2)
        print_results(command.replace('|', ' + '), common)

    # elif command.startswith('~'):  # слова через запятую как вектор ассоциаций
    #     results = model.get_top_similar_for_assoc(command[1:].split(','), count)
    #     print_results(command, results)

    else:  # поиск похожих. Слово или слова через запятую (столица,украина; король,-мужчина, женщина)
        results = model.get_top_similar_to_words(command.split(','), count)
        print_results(command, results)


def print_results(title, results):
    print(title)
    print ("=" * 20)
    
    for word_name, similarity in results:    
        print("{:0.3f}\t{}".format(similarity, word_name))

    print("")
          

def interactive(model, count=15):
    while True:
        print('\nВведите слово или нажмите [Enter] для показа случайного.\nДля выхода введите [q] или [x].')
        command = input().lower() or get_random_command(model)

        if command in {'q', 'x'}:
            return

        try:
            do_command(model, command, count)
        except KeyError:
            print ("Слово не найдено\n")
            

def get_random_command(model):
    word = model.get_random_word()
    words = [word]
    words_count = choice([1,1,1,1,1,2,2,2,3])
    if words_count > 1:
        for other_word in take_n_from_top_similar(model, word, words_count - 1):
            sign = choice(['', '', '-'])
            if sign != '-':
                if choice([False, False, True]):
                    other_word = model.get_random_word()
            words.append(sign + other_word)

    return ",".join(words)


def take_n_from_top_similar(model, word_name, n, top_n=5):
    results = model.get_top_similar_to_words([word_name], top_n)
    words = {word_name for word_name, similarity in results}
    for _ in range(n):
        if words:
            word = choice(list(words))
            words.remove(word)
            yield word
        else:
            break


if __name__ == '__main__':
    args = sys.argv[1:]
    path, command, count, is_interactive = MODEL_PATH, None, 15, False

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

    main(SOURCE_PATH, path, command, count, is_interactive)