import faiss
import numpy as np
import torch
import tables
import os

def to_onehot(x, c):
    bsz = x.size(0)
    x_onehot = torch.FloatTensor(bsz, c)
    x_onehot.zero_()
    x_onehot.scatter_(1, x, 1)
    return x_onehot

def safe_exp(x):
    max_x, _ = torch.max(x, 1, keepdim = True)
    x = x - max_x
    return torch.exp(x)

def retrive_items(table, idx):
    return np.take(table, idx, axis=0)

def get_knnp(hidden, index, topk, target_table, vocab_size):
    device = hidden.device
    hidden = hidden.contiguous().view(-1, hidden.size(-1)).cpu()
    seq_len = hidden.size(0)
    D_, I = index.search(hidden.numpy(), topk)
    idx = retrive_items(target_table, I).squeeze(-1)
    idx = torch.LongTensor(idx).to(device)
    D = torch.FloatTensor(D_).to(device)
    D = D.sqrt()
    D = safe_exp(-D)
    #print(D[0])
    #print(idx[0])
    knnp = sum_over_topk(idx, D, vocab_size)

    knnp_sum = torch.sum(knnp, -1, keepdim=True)
    knnp = knnp/knnp_sum
    return knnp

def calc_L2(hidden, retrived):
    bsz = retrived.size(0)
    hidden = hidden.view(bsz, 1, -1)
    D = torch.pow(hidden - retrived, 2)
    D = D.sum(-1).sqrt()
    return D

def calc_dot(hidden, retrived):
    bsz = retrived.size(0)
    hidden = hidden.view(bsz, 1, -1)
    D = (hidden * retrived).sum(-1)
    return D
    
def sum_over_topk(idx, D, vocab_size):
    p = torch.zeros(D.size(0), vocab_size+1, device=D.device)
    fake_token = vocab_size
    # sort and get cumsum
    sorted_idx, indices = torch.sort(idx, 1)
    sorted_D = torch.gather(D, 1, indices)
    sorted_D = sorted_D.cumsum(1)
    mask = torch.zeros(D.size(), device=D.device).byte()
    mask[:, :-1] = sorted_idx[:, :-1] == sorted_idx[:, 1:]
    mask = mask.bool()
    sorted_idx = sorted_idx.masked_fill(mask, fake_token) 
    sorted_D = sorted_D.masked_fill(mask, 0)
    # remove cumsum by sorting
    idx_ = torch.arange(D.size(1), device=D.device).unsqueeze(0)
    idx_ = idx_.expand(D.size(0), D.size(1)).masked_fill(mask, -1)

    _, indices = torch.sort(idx_, 1)
    sorted_idx = torch.gather(sorted_idx, 1, indices)

    sorted_D = torch.gather(sorted_D, 1, indices)
    sorted_D[:, 1:] = sorted_D[:, 1:] - sorted_D[:, :-1]
    p = p.scatter_(1, sorted_idx, sorted_D)
    return p[:, :-1]



def read_target_table():
    print('loading target table')
    #return np.zeros((10000, 1), dtype=int)
    d = '/home/laiguokun/ssd/record/'
    target = [] 
    for i in range(8):
        fn = os.path.join(d, 'target_{}.h5'.format(i))
        tmp = 50 * 1000 * 1000
        with tables.open_file(fn, 'r') as fin:
            data = fin.root.data[:tmp, :]
            target.append(data)
    target = np.concatenate(target, axis=0)
    print('target table shape:{}'.format(target.shape))
    return target