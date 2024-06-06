import os
import torch
import pickle
from torch.nn.utils.rnn import PackedSequence

from Dataset import CustomDataset, Operator


def load_dataset(path):
    # assemble dataset from directory
    dps = []
    labels = []

    op_name = [op.name for op in list(Operator)]

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        with open(file_path, 'rb') as file:
            local_dps, local_label = pickle.load(file)

            assert local_label in op_name
            label_idx = op_name.index(local_label)

            dps.extend(local_dps)
            labels.extend([label_idx] * len(local_dps))

    assert len(dps) == len(labels)
    
    dataset = CustomDataset(dps, labels)
    return dataset 


def unpack_sequence(packed_sequence, lengths):
    assert isinstance(packed_sequence, PackedSequence)
    head = 0
    trailing_dims = packed_sequence.data.shape[1:]
    unpacked_sequence = [torch.zeros(l, *trailing_dims) for l in lengths]
    # l_idx - goes from 0 - maxLen-1
    for l_idx, b_size in enumerate(packed_sequence.batch_sizes):
        for b_idx in range(b_size):
            unpacked_sequence[b_idx][l_idx] = packed_sequence.data[head]
            head += 1
    return unpacked_sequence
