import pickle
import os


def load_func(path_rel_to_load):

    this_dir, this_filename = os.path.split(__file__)
    data_path = os.path.join(this_dir, path_rel_to_load)
    data = pickle.load(open(data_path, 'rb'))

    return data
