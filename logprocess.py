#%%
import re

def remove_chars(line_array):
    new_words = []
    for word in line_array:
        word = word.strip()
        word = re.sub(r'[^a-zA-ZäöÄÖ ]+', '', word)
        if word != '':
            new_words.append(word)
    return new_words


def clean_line(line):
    line = line.split()[3:]
    return remove_chars(line)

def get_lines(filepath):
    lines = []
    with open(filepath, encoding="utf8") as fp:
        for line in fp:
            line = clean_line(line)
            lines.append(line)
    return lines

#%%
def build_vocab(tokenized_corpus):
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    return vocabulary, word2idx, idx2word

#%%
def build_bag_and_vocab(filepath='./testlog.txt'):
    tokens = get_lines(filepath)
    vocab, w2i, i2w = build_vocab(tokens)

    import numpy as np
    vocab = np.array(vocab)

    return tokens, vocab, w2i, i2w



#%%

tokens, vocab, w2i, i2w = build_bag_and_vocab()

#%%
w2i['öö']

#%%
