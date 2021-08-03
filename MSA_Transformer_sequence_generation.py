import sys
import numpy as np
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
from tqdm import tqdm
import random

import esm
import torch

# command line: D:/Deep_Physics/Protein/carnevale-lab/small_potts_datasets/RRM_VAE_paper/train10k.txt MSATr_entropy_10k 10000 4 256 32

file_name = sys.argv[1] # train10k.txt'
save_name = sys.argv[2]
n_generate = int(sys.argv[3])
n_stack = int(sys.argv[4]) 
n_batch = int(sys.argv[5])
n_mask = int(sys.argv[6])
save_interval = int(sys.argv[7])

torch.set_printoptions(precision=3, sci_mode=False)
torch.set_grad_enabled(False)

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def loaded_msa(msa, nseq) -> List[Tuple[str, str]]:
    """ Reads the nseq sequences at constant intervals from an MSA file, automatically removes insertions."""
    N = len(msa)
    # split into chunks of approximately equal lengths (https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length)
    splits = np.array_split(range(1,N), nseq)
    output = []
    for spl in splits:
        record = msa[spl[0]]
        output.append( (record['description'], remove_insertions(str(record['seq']))) )
    return output

def loaded_msa_all(msa) -> List[Tuple[str, str]]:
    """ Reads the nseq sequences at constant intervals from an MSA file, automatically removes insertions."""
    N = len(msa)
    return [(record['description'], remove_insertions(str(record['seq'])))
            for record in itertools.islice(msa, 1, N)]

# https://stackoverflow.com/questions/57237596/how-to-improve-np-random-choice-looping-efficiency
def vectorized_choice(p, n, items=None):
    s = p.cumsum(axis=1)
    r = np.random.rand(p.shape[0], n, 1)
    q = np.expand_dims(s, 1) >= r
    k = q.argmax(axis=-1)
    if items is not None:
        k = np.asarray(items)[k]
    return k


""" prepare Protein data """
Lines = open(file_name, 'r').readlines()
seqs = [l.replace('\n', '') for l in Lines if len(l) > 3]
msa = [{'description': 'noname', 'seq': seq} for seq in seqs]
msa_data = loaded_msa_all(msa)


alphabet = 'ACDEFGHIKLMNPQRSTVWY-'
alph_dict = {}
for i, a in enumerate(alphabet):
    alph_dict[a] = i

M = len(seqs)
L = len(seqs[0])
A = len(alphabet)
one_hot = np.zeros((M, L, A))
for m in range (M):
    for i in range (L):
        one_hot[m, i, alph_dict[seqs[m][i]]] = 1

counts = one_hot.sum(0)
indep = counts / counts.sum(-1).reshape( (-1, 1) )
entropy_per_pos = (- indep * np.log(indep+1e-9)).sum(-1)
pos_seq = (entropy_per_pos.argsort() + 1).tolist() # +1 to correct for start token



msa_transformer, msa_alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
msa_transformer = msa_transformer.eval().cuda()
msa_batch_converter = msa_alphabet.get_batch_converter()



standard_idx = [msa_alphabet.get_idx(tok) for tok in alphabet]
L = len(msa_data[0][1]) + 1 # add start token
A = len(standard_idx)
new_seq = []
for i in tqdm(range(0, n_generate, n_mask*n_stack)):
    msa_batch_data = []
    for s in range(n_stack):
        idxs = random.sample(range(len(msa_data)), n_batch)
        msa_batch_data.append( [msa_data[i] for i in idxs] )
    
    msa_batch_labels, msa_batch_strs, msa_batch_tokens =\
        msa_batch_converter(msa_batch_data)
    msa_batch_tokens = msa_batch_tokens.cuda()
    
    new_tokens = msa_batch_tokens.clone()
    
    prot_idxs = np.random.randint(n_batch, size=n_mask)
    new_tokens[:, prot_idxs, 1:] = msa_alphabet.mask_idx # mask certain proteins entirely (except start token)
    # generate {batch_size} samples, one position at a time
    for pos in pos_seq:
        # run model and gather masked probabilities          
        output = msa_transformer(new_tokens)
        probs = torch.softmax(output['logits'][:, prot_idxs][:, :, pos, standard_idx], -1).detach().cpu().numpy()
        # sample random tokens based on predicted probabilities (Gibbs sampling)
        rand_res = vectorized_choice(probs.reshape((n_stack*n_mask, A)), 1).flatten()
        toks = [standard_idx[t] for t in rand_res]
        toks = torch.tensor(toks).reshape((n_stack, n_mask)).cuda()
        # replace mask with samples
        idxs_scat = torch.tensor(prot_idxs, dtype=int).cuda().expand(n_stack, -1)
        new_tokens[:,:,pos].scatter_(1, idxs_scat, toks)
    new_tokens = new_tokens.detach().cpu().numpy()
    new_seq.append( new_tokens[:, prot_idxs, 1:].reshape((-1, L-1)) ) # drop start token
    
    if len(new_seq)*n_stack*n_mask > save_interval:
        new_seq = np.concatenate(new_seq)
        new_strs = []
        for seq in new_seq:
            chars = [msa_alphabet.get_tok(idx) for idx in seq]
            new_strs.append(''.join(chars))
        
        with open(save_name + '.txt', 'a') as file_handler:
            for item in new_strs:
                file_handler.write("{}\n".format(item))
        new_seq = []
        
new_seq = np.concatenate(new_seq)
new_strs = []
for seq in new_seq:
    chars = [msa_alphabet.get_tok(idx) for idx in seq]
    new_strs.append(''.join(chars))

with open(save_name + '.txt', 'a') as file_handler:
    for item in new_strs:
        file_handler.write("{}\n".format(item))
new_seq = []