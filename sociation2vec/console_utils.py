# coding: utf-8
def print_results(title, results):
    print(title)
    print ("=" * 20)
    
    for word_name, similarity in results:    
        print("{:0.3f}\t{}".format(similarity, word_name))

    print("")


def query_model_print_results(model, query, count=10):
    results = model.get_top_similar_to_words(query.lower().split(','), count=count)
    print_results(query, results)