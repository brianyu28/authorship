from itertools import dropwhile

def prune_counter(counter, threshold):
    for key, count in dropwhile(lambda key_count: key_count[1] >= threshold, counter.most_common()):
        del counter[key]

def is_training(identifier):
    return identifier[0] == 0
