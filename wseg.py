#!/usr/bin/python
# -*- coding: utf-8 -*-
from vncorenlp import VnCoreNLP
import pandas as pd


def concat(items: list) -> str:
    if type(items[0]) == list:
        # nested list
        _items = [item for nested in items for item in nested]
        return " ".join(_items)
    else:
        return " ".join(items)

def simple_usage():
    # Uncomment this line for debugging
    # logging.basicConfig(level=logging.DEBUG)

    vncorenlp_file = r'./VnCoreNLP/VnCoreNLP-1.1.1.jar'
    df = pd.read_csv('data/test.csv')
    vi_sents = df['vi'].values
    print("Load data successfully")
    print(vi_sents[0:3])

    # Use "with ... as" to close the server automatically
    with VnCoreNLP(vncorenlp_file) as vncorenlp:
        vi_wseg_sents = [concat(vncorenlp.tokenize(sent)) for sent in vi_sents]
        df['vi_wseg'] = vi_wseg_sents
    print("Segment successfully")
    df.to_csv('data/test_wseg.csv')

if __name__ == '__main__':
    simple_usage()