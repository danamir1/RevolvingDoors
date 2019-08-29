import pickle

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def read_list_from_csv(path, key='term', sep=','):
    with open(path, 'r', encoding='utf8') as f:
        keys = f.readline().strip().split(sep)
        key_index = keys.index(key)
        terms_list = []
        for line in f.readlines():
            try:
                terms_list.append(line.strip().split(sep)[key_index])
            except:
                continue
    return terms_list