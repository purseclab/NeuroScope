from enum import Enum
from typing import NamedTuple
import random
from torch.nn.utils.rnn import pad_sequence


import torch
from torch.utils.data import IterableDataset, Dataset

from gen import *
from range import *


class OpType:
    """
    The weight when randomly choosing the optype
    """

    op_cnt = 0

    def __init__(self, name, weight, gen_f):
        self.id = OpType.op_cnt
        OpType.op_cnt += 1
        self.name = name
        self.weight = weight
        self.gen_f = gen_f


class Operator(Enum):
    @classmethod
    def weights(self):
        return [op.weight for op in list(self)]

    @property
    def weight(self):
        return self.value.weight

    def get_dp(self, batch_size):
        return self.value.gen_f(batch_size)

    def get_label(self):
        return self.value.id

    CONV = OpType("CONV", 1, conv_gen)
    CONV_RELU = OpType("CONV_RELU", 1, conv_relu_gen)
    # CONV_BN = OpType('CONV_BN', 1, conv_bn_gen)

    FC = OpType("FC", 1, fc_gen)

    AVG_POOL = OpType("AVG_POOL", 1, avg_pool_gen)
    MAX_POOL = OpType("MAX_POOL", 1, max_pool_gen)

    RELU = OpType("RELU", 1, relu_gen)
    # PRELU = OpType('PRELU', 1, prelu_gen)
    # ELU = OpType('ELU', 1, elu_gen)

    # BN = OpType('BN', 1, batch_norm_gen)

    SOFTMAX = OpType("SOFTMAX", 1, softmax_gen)
    # SIGMOID = OpType('SIGMOID', 1, sigmoid_gen)

    # FLATTEN = OpType('FLATTEN', 1, flatten_gen)
    # SQUEEZE = OpType('SQUEEZE', 1, squeeze_gen)
    # UNSQUEEZE = OpType('SQUEEZE', 1, unsqueeze_gen)

    ADD = OpType("ADD", 1, add_gen)
    
    # MUL = OpType("MUL", 1, mul_gen)

    TRANSPOSE = OpType('TRANSPOSE', 1, transpose_gen)
    # RESHAPE = OpType('RESHAPE', 1, reshape_gen)

    # CONCAT = OpType("CONCAT", 1, concat_gen)
    
    # ARGMAX = OpType("ARGMAX", 1, argmax_gen)


def get_dp(multiple_io_sample=None):
    """
    Generate one datapoint and return.
    If multiple_io_sample, each generated datapoint contains multiple io sample
    """
    # choose one op type according to the pre-defined weights
    op = random.choices(list(Operator), weights=Operator.weights())[0]

    if multiple_io_sample is not None:
        batch_size = multiple_io_sample
    else:
        batch_size = 1

    # synthesize one dp
    while True:
        try:
            dps = op.get_dp(batch_size)
            break
        except:
            print(op.name + " except")
            pass

    # datapoints (io sample) and their labels
    return (dps, op.name)


class CustomDataset(Dataset):
    def __init__(self, dps, labels):
        assert isinstance(dps, list)
        assert isinstance(labels, list)
        assert len(dps) == len(labels)
        
        self.dps = dps
        self.labels = labels
        
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.dps[idx], self.labels[idx]

    
def collate_fn(batch):
    data = [[item[0].input_ftr, item[0].output_ftr, item[0].features()] for item in batch]
    
    input_sequence_data = [item[0].input_ftr for item in batch]
    input_lengths = [len(seq) for seq in input_sequence_data]
    padded_input_sequence_data = pad_sequence(input_sequence_data, batch_first=True, padding_value=0)
    
    output_sequence_data = [item[0].output_ftr for item in batch]
    output_lengths = [len(seq) for seq in output_sequence_data]
    padded_output_sequence_data = pad_sequence(output_sequence_data, batch_first=True, padding_value=0)
    
    feature_data = torch.tensor([item[0].features() for item in batch])
    
    label = torch.tensor([item[1] for item in batch])
    
    return (padded_input_sequence_data, input_lengths), (padded_output_sequence_data, output_lengths), feature_data, label
    

class InMemDataset(IterableDataset):
    """
    Synthesize tons of datapoints beforehand and store in memory.
    It provide __next__ to create an illusion that it synthesize "on-the-fly"
    """

    def __init__(self, size, multiple_io_sample=False):
        self.size = size

        # internal index, keeping track of where we have iterated to
        self._idx = 0

        # list of indices
        self._shuffle_list = list(range(size))

        self._dataset = []
        for _ in range(self.size):
            self._dataset.append(get_dp(multiple_io_sample))

    def __iter__(self):
        return self

    def __next__(self):
        """
        Choose the next one from the pre-generated list
        """
        dp = self._dataset[self._shuffle_list[self._idx]]
        self._idx += 1
        if self._idx == self.size:
            self._idx = 0
            self._reshuffle()
        return dp

    def _reshuffle(self):
        """
        Reshuffle the dataset
        """
        random.shuffle(self._shuffle_list)

    def num_label(self):
        return len(list(Operator))

    def size(self):
        return self.size


class OTFDataset(IterableDataset):
    """
    Synthesize data on-the-fly. No datapoint is used twice.
    """

    def __init__(self) -> None:
        pass

    def __iter__(self):
        return self

    def __next__(self):
        """
        Synthesize one datapoint
        """
        return get_dp()

    def num_label(self):
        return len(list(Operator))


class TestDataset:
    """
    TODO:
    It is NOT a pytorch dataset.
    It tests the accuracy per operator type
    """

    def __init__(self) -> None:
        pass

    def test(self, model):
        pass


def get_batch_from_dataset(dataset, size):
    """
    This is for OTFDataset
    """
    input_ftrs = []
    output_ftrs = []
    input_lens = []
    output_lens = []
    labels = []
    for dp_id, dp in enumerate(dataset):
        if dp_id == size:
            break

        input_ftrs.append(dp.input_ftr)
        input_lens.append(dp.input_len)
        output_ftrs.append(dp.output_ftr)
        output_lens.append(dp.output_len)
        labels.append(dp.label)

    return input_ftrs, input_lens, output_ftrs, output_lens, labels


def gen_batch_from_dataset(dataset, size):
    """
    This is for OTFDataset
    """
    input_ftrs = []
    output_ftrs = []
    input_lens = []
    output_lens = []
    labels = []
    for dp_id, dp in enumerate(dataset):
        if dp_id == size:
            break

        input_ftrs.append(dp.input_ftr)
        input_lens.append(dp.input_len)
        output_ftrs.append(dp.output_ftr)
        output_lens.append(dp.output_len)
        labels.append(dp.label)

    return input_ftrs, input_lens, output_ftrs, output_lens, labels


def old_collate_fn(data):
    """
    In this function, we do two things:
        1. concatenate input and output
        2. pad

    data: (input, output, target_label)
    """
    # max_len
    max_len = max([dp.len for dp in data])

    # init features: (batch_size, max_len)
    features = torch.zeros(len(data), max_len)

    # concat with padding
    for idx, dp in enumerate(data):
        features[idx] = torch.cat([dp.input, dp.output, torch.zeros(max_len - dp.len)])

    # label
    labels = torch.tensor([dp.label for dp in data])

    # lengths
    lengths = torch.tensor([dp.len for dp in data])

    return features, labels, lengths
