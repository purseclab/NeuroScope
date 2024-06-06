import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence
import torch.nn.init as init

from utils import unpack_sequence

import logging

from IPython import embed


class HybridNet(nn.Module):
    def __init__(self, ftr_len):
        super().__init__()
        
        self.hidden_size = 16
        self.bidirectional = False
        
        self.lstm1 = nn.LSTM(
            1, self.hidden_size, batch_first=True, bidirectional=self.bidirectional
        )
        self.lstm2 = nn.LSTM(
            1, self.hidden_size, batch_first=True, bidirectional=self.bidirectional
        )
        
        self.fc1 = nn.Linear(2*self.hidden_size + ftr_len, 8)
        self.fc2 = nn.Linear(8, 1)
    
    def forward(self, 
                x,
                x_lens,
                y,
                y_lens,
                feature_data):
        
    
        # Process input sequence 
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        o_n_x, (h_n_x, _) = self.lstm1(x_packed)

        # Process output sequence
        y_packed = torch.nn.utils.rnn.pack_padded_sequence(y, y_lens, batch_first=True, enforce_sorted=False)
        o_n_y, (h_n_y, _) = self.lstm2(y_packed)

        # Concatenate outputs
        combined_feature = torch.cat((h_n_x.squeeze(0), h_n_y.squeeze(0), feature_data), dim=1)

        out = torch.relu(self.fc1(combined_feature))
        out = self.fc2(out)
        return out
        


class FeatureNet(nn.Module):
    def __init__(self, ftr_len, num_label):
        super().__init__()
        self.fc1 = nn.Linear(ftr_len, 128)  # Adjust input size based on features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, num_label)  # Adjust output size based on labels

    def forward(self, 
                x,
                x_lens,
                y,
                y_lens,
                feature_data):
        # only use feature_data here
        out = torch.relu(self.fc1(feature_data))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out))
        out = self.fc5(out)
        return out

    
class Seq2seq(nn.Module):
    """
    Ref: https://github.com/rantsandruse/pytorch_lstm_03classifier
    """

    def __init__(self, hidden_size, output_size, device, bidirectional=False):
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # (input_size, )
        # input_size = 1: each input size is 1 (one floating point)
        self.input_lstm = nn.LSTM(
            1, self.hidden_size, batch_first=True, bidirectional=bidirectional
        )
        self.output_lstm = nn.LSTM(
            1, self.hidden_size, batch_first=True, bidirectional=bidirectional
        )

        self.hidden2label = nn.Linear(self.hidden_size, output_size)

    def forward(self, dps):
        # one op with multiple io sample
        input_ftrs = [dp.input_ftr for dp in dps]
        # input_lens should be all the same
        input_lens = [dp.input_len for dp in dps]

        output_ftrs = [dp.output_ftr for dp in dps]
        # output_lens should be all the same
        output_lens = [dp.output_len for dp in dps]

        # input_lstm
        packed_input_ftrs = pack_sequence(input_ftrs, enforce_sorted=False).to(
            self.device
        )
        input_lstm_out, input_hidden = self.input_lstm(packed_input_ftrs)
        input_lstm_out, input_lstm_out_lens = pad_packed_sequence(
            input_lstm_out, batch_first=True, total_length=max(input_lens)
        )

        # output_lstm: use the output_ftrs and the input_hidden
        packed_output_ftrs = pack_sequence(output_ftrs, enforce_sorted=False).to(
            self.device
        )
        output_lstm_out, hidden = self.output_lstm(packed_output_ftrs, input_hidden)
        output_lstm_out, output_lstm_out_lens = pad_packed_sequence(
            output_lstm_out, batch_first=True, total_length=max(output_lens)
        )

        # we only use the last output
        if self.bidirectional:
            # Sum bidirectional LSTM outputs
            lstm_out_forward = (
                output_lstm_out[:, -1, : self.hidden_size]
                + output_lstm_out[:, -1, self.hidden_size :]
            )
        else:
            lstm_out_forward = output_lstm_out[:, -1]

        # linear layer
        label_space = self.hidden2label(lstm_out_forward)

        # if torch.isnan(label_space).any():
        # embed()

        return label_space


class AttentionalSeq2seq(nn.Module):
    """
    We are using GRU instead of LSTM, for easier applying attention to the hidden state (LSTM has extra cell state)

    Ref: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html?highlight=chat%20bot
    """

    def __init__(self, hidden_size, output_size, device, bidirectional=False):
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # (input_size, )
        # input_size = 1: each input size is 1 (one floating point)
        self.input_lstm = nn.GRU(
            1, self.hidden_size, batch_first=True, bidirectional=bidirectional
        )

        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.output_lstm = nn.GRU(
            1, self.hidden_size, batch_first=True, bidirectional=bidirectional
        )

        self.hidden2label = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_ftrs, input_lens, output_ftrs, output_lens):
        # input_lstm
        input_ftrs = pack_sequence(input_ftrs, enforce_sorted=False).to(self.device)
        input_lstm_out, input_hidden = self.input_lstm(input_ftrs)
        input_lstm_out, input_lstm_out_lens = pad_packed_sequence(
            input_lstm_out, batch_first=True, total_length=max(input_lens)
        )

        # assume all output_ftrs have the same lens
        assert all([length == output_lens[0] for length in output_lens])
        output_len = output_lens[0]

        output_ftrs = torch.stack(output_ftrs).to(self.device)
        # print("output_ftrs shape: ", output_ftrs.shape)
        # print("output_ftrs: ", output_ftrs)

        # use the last input_hidden as the init hidden
        output_lstm_hidden = input_hidden

        output_lstm_out_at_t = None

        print(input_lens[0] + output_lens[0])

        # output_lstm loop
        for t in range(output_len):
            output_ftr_at_t = torch.index_select(
                output_ftrs, 1, torch.tensor([t]).to(self.device)
            )
            # print("output_ftr_at_t shape: ", output_ftr_at_t.shape)
            # print("output_ftr_at_t: ", output_ftr_at_t)

            # lstm at t
            output_lstm_out_at_t, output_lstm_hidden = self.output_lstm(
                output_ftr_at_t, output_lstm_hidden
            )
            # print("output_lstm_out_at_t shape", output_lstm_out_at_t.shape)
            # print("output_lstm_hidden shape", output_lstm_hidden.shape)

            # calculate attention weights
            attn_energies = torch.sum(input_lstm_out * self.attn(input_lstm_out), dim=2)
            # print("attn_energies shape: ", attn_energies.shape)
            attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)
            # print("attn_weights shape: ", attn_weights.shape)

            # multiply attention weights to input_lstm_out to get new "weighted sum" context vector
            context = attn_weights.bmm(input_lstm_out).squeeze(1)
            # print("context shape: ", context.shape)

            # concatenate weighted context vector and GRU hidden state
            output_lstm_hidden = output_lstm_hidden.squeeze(0)
            # print("output_lstm_hidden shape: ", output_lstm_hidden.shape)
            concat_input = torch.cat((output_lstm_hidden, context), 1)
            # print("concat_input shape: ", concat_input.shape)
            # it is used as the next hidden state
            output_lstm_hidden = self.concat(concat_input).unsqueeze(0)
            # print("final output_lstm_hidden shape: ", output_lstm_hidden.shape)

            del output_ftr_at_t, attn_energies, attn_weights, context, concat_input

            torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated())

        print(torch.cuda.memory_allocated())
        print("----------------")

        # we only use the last output
        if self.bidirectional:
            # Sum bidirectional LSTM outputs
            lstm_out_forward = (
                output_lstm_out_at_t[:, :, : self.hidden_size]
                + output_lstm_out_at_t[:, :, self.hidden_size :]
            )
            lstm_out_forward = lstm_out_forward.squeeze(1)
        else:
            lstm_out_forward = output_lstm_out_at_t.squeeze(1)

        # linear layer
        label_space = self.hidden2label(lstm_out_forward)

        # very easy OOM
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated())

        return label_space
