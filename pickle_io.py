import cPickle as pickle

def pickle_save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def pickle_load(path):
    with open(path) as f:
        obj = pickle.load(f)
    return obj
