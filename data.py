def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


d1 = unpickle("data/data_batch_1")
print(len(d1))