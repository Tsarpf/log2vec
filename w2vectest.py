#%%
import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F



#  #%%
#  corpus = [
#      'he is a king',
#      'she is a queen',
#      'he is a man',
#      'she is a woman',
#      'warsaw is poland capital',
#      'berlin is germany capital',
#      'paris is france capital',   
#  ]
#  
#  #%%
#  def tokenize_corpus(corpus):
#      tokens = [x.split() for x in corpus]
#      return tokens
#  
#  tokenized_corpus = tokenize_corpus(corpus)
#  
#  #%%
#  vocabulary = []
#  for sentence in tokenized_corpus:
#      for token in sentence:
#          if token not in vocabulary:
#              vocabulary.append(token)
#  
#  word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
#  idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

from logprocess import build_bag_and_vocab
tokenized_corpus, vocabulary, word2idx, idx2word = build_bag_and_vocab()

vocabulary_size = len(vocabulary)
#%%
window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, treated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array

#%%
def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x
#%%
embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 101
learning_rate = 0.001

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    #if epo % 10 == 0:    
    print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')


#%%
W1numpy = torch.Tensor.cpu(W1).detach().numpy()
W2numpy = torch.Tensor.cpu(W2).detach().numpy()
#%%
def cosine_distance(u, v):
 return np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v))

#%%
word_idxs = []
for idx,_ in enumerate(idx2word):
    word_idxs.append(idx2word[idx])
#%%
from sklearn.metrics.pairwise import cosine_similarity

def closest(vec, array):
    sims = cosine_similarity([vec], array)[0]
    maxed = np.where(sims == np.max(sims))[0][0]
    return idx2word[maxed]

def closest_to_word(word, array):
    idx = word2idx[word]
    array_remo = np.delete(array, idx, 0)
    return closest(array[idx], array_remo)
#%%
embeddings = np.transpose(W1numpy)
#%%
for idx,_ in enumerate(idx2word):
    word = idx2word[idx]
    print('%s -> %s' % (word, closest_to_word(word, embeddings)))
#%%

def build_mat(sims):
    dtype = [('word', np.unicode_, 16), ('similarity', float)]
    values = []
    for idx, _ in enumerate(sims):
        word = idx2word[idx]
        values.append((word, sims[idx]))
    return np.array(values, dtype=dtype)

#%%
def closest_mat(vec, array):
    sims = cosine_similarity([vec], array)[0]
    mat = build_mat(sims)
    return np.sort(mat, order='similarity')[::-1]


#%%
for idx,_ in enumerate(idx2word):
    word = idx2word[idx]
    print('%s -> %s' % (word, closest_mat(embeddings[idx], embeddings)))

#%%
