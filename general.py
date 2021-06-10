def get_vocab_dict(items):  # 用于编号
    item2idx = {}
    idx = 0
    for item in items:
        item2idx[item] = idx
        idx += 1
    return item2idx
