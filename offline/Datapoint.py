import torch
import random
import time

class DataPoint:
    """
    Datapoint representation. 
    We assume there at most two weights: weight_0, weight_1. 
    """

    def __init__(self, input, output, num_input=1, weight_0=None, weight_1=None, info=None):
        # the _raw_* have shape info
        self._raw_input = input
        self._raw_output = output
        self._raw_weight_0 = weight_0
        self._raw_weight_1 = weight_1

        # self.input/self.output are the flatten version of "raw" input/output
        self.input = self._raw_input.reshape(-1)
        self.output = self._raw_output.reshape(-1)

        # other info, for instance, kernel shape
        self.info = info
        self.label = None
        
        # input stat features
        self.input_min = min(self.input).item()
        self.input_max = max(self.input).item()
        self.input_avg = torch.mean(self.input).item()
        self.input_zero_count = torch.count_nonzero(self.input == 0).item()
        if len(self.input) == 1:
            self.input_var = 0
        else:
            self.input_var = torch.var(self.input)
        self.num_input = num_input
        
        # output stat features
        self.output_min = min(self.output).item()
        self.output_max = max(self.output).item()
        self.output_avg = torch.mean(self.output).item()
        self.output_zero_count = torch.count_nonzero(self.output == 0).item()
        if len(self.output) == 1:
            self.output_var = 0
        else:
            self.output_var = torch.var(self.output)
        
        # input/output feature
        self.input_output_len_ratio = self.input_len / self.output_len
        if self.output_avg == 0:
            self.input_output_avg_ratio = 0
        else:
            self.input_output_avg_ratio = self.input_avg / self.output_avg
        if self.output_min == 0:
            self.input_output_min_ratio = 0
        else:
            self.input_output_min_ratio = self.input_min / self.output_min
        if self.output_max == 0:
            self.input_output_max_ratio = 0
        else:
            self.input_output_max_ratio = self.input_max / self.output_max
        if self.output_var == 0:
            self.input_output_var_ratio = 0
        else:
            self.input_output_var_ratio = self.input_var / self.output_var
        
        # weight stat features
        if weight_0 is None:
            self.num_weight = 0
        elif weight_1 is None:
            self.num_weight = 1
        else:
            self.num_weight = 2
            
        if weight_0 is None:
            self.weight_0_len = 0
            self.input_weight_0_len_ratio = 0
        else:
            self.weight_0_len = weight_0.numel()
            self.input_weight_0_len_ratio = self.input_len / self.weight_0_len                
        if weight_1 is None:
            self.weight_1_len = 0
            self.input_weight_1_len_ratio = 0
        else:
            self.weight_1_len = weight_1.numel()
            self.input_weight_1_len_ratio = self.input_len / self.weight_1_len
            
        if weight_0 is not None and weight_1 is not None:
            self.weight_0_weight_1_len_ratio = self.weight_0_len / self.weight_1_len
        else:
            self.weight_0_weight_1_len_ratio = 0

            
    
    def recompute(self):
        """
        We add some features after the dataset is generated, 
        so the additional features need to be computed. 
        """
        if self._raw_weight_0 is None:
            self.weight_0_len = 0
            self.input_weight_0_len_ratio = 0
        else:
            self.weight_0_len = self._raw_weight_0.numel()
            self.input_weight_0_len_ratio = self.input_len / self.weight_0_len                
        if self._raw_weight_1 is None:
            self.weight_1_len = 0
            self.input_weight_1_len_ratio = 0
        else:
            self.weight_1_len = self._raw_weight_1.numel()
            self.input_weight_1_len_ratio = self.input_len / self.weight_1_len
            
        if self._raw_weight_0 is not None and self._raw_weight_1 is not None:
            self.weight_0_weight_1_len_ratio = self.weight_0_len / self.weight_1_len
        else:
            self.weight_0_weight_1_len_ratio = 0
        
        if self.output_avg == 0:
            self.input_output_avg_ratio = 0
        else:
            self.input_output_avg_ratio = self.input_avg / self.output_avg
        if self.output_min == 0:
            self.input_output_min_ratio = 0
        else:
            self.input_output_min_ratio = self.input_min / self.output_min
        if self.output_max == 0:
            self.input_output_max_ratio = 0
        else:
            self.input_output_max_ratio = self.input_max / self.output_max
        if self.output_var == 0:
            self.input_output_var_ratio = 0
        else:
            self.input_output_var_ratio = self.input_var / self.output_var
        

    def set_label(self, label):
        self.label = label

    @property
    def len(self):
        """
        Return the total len
        """
        return self.input_len + self.output_len

    @property
    def input_ftr(self):
        # we need to have shape as (timestamp, 1),
        # where our input_size (size of input at each timestamp) is 1
        return self.input.view(-1, 1)

    @property
    def output_ftr(self):
        return self.output.view(-1, 1)

    @property
    def input_len(self):
        assert len(self.input.shape) == 1
        return self.input.size()[0]

    @property
    def output_len(self):
        assert len(self.output.shape) == 1
        return self.output.size()[0]
    
    def features(self):
        return [
                # input
                self.num_input,
                self.input_len,
                self.input_min,
                self.input_max,
                self.input_avg,
                self.input_var,
                self.input_zero_count,
                
                # output
                self.output_len,
                self.output_min, 
                self.output_max,
                self.output_avg,
                self.output_var,
                self.output_zero_count,
                
                # input/output ratio
                self.input_output_len_ratio,
                self.input_output_avg_ratio, 
                self.input_output_min_ratio,
                self.input_output_max_ratio,
                self.input_output_var_ratio,
                        
                # weight
                self.num_weight,
                self.weight_0_len,
                self.input_weight_0_len_ratio,
                self.weight_1_len,
                self.input_weight_1_len_ratio,
                self.weight_0_weight_1_len_ratio
               ]
    
    def features_len(self):
        return len(self.features())

    """
    @property
    def input_min(self):
        return min(self.input)
    
    @property
    def output_min(self):
        return min(self.output)
    
    @property
    def input_max(self):
        return max(self.input)
    
    @property
    def output_max(self):
        return max(self.output)
    
    @property 
    def input_avg(self):
        return torch.mean(self.input)
    
    @property
    def output_avg(self):
        return torch.mean(self.output)
    
    @property 
    def input_zero_count(self):
        return torch.count_nonzero(self.input == 0).item()
    
    @property 
    def output_zero_count(self):
        return torch.count_nonzero(self.output == 0).item()
    """

    def __str__(self):
        ret_str = ""
        for attr_name, attr_value in self.__dict__.items():
            ret_str += f"{attr_name}: {attr_value}\n"
        return ret_str


    def _aot(self):
        """
        Deprecated
        Transform input/output tensor to aot-like structure
        """
        assert False
        self.input = self._raw_input.view(-1)
        self.output = self._raw_output.view(-1)

    def _tflm(self):
        """
        Deprecated
        Transform input/output tensor to tflm-like structure
        """
        assert False
        # (num_dim, size_of_each_dim, type, data)
        self.input = torch.cat(
            (
                torch.tensor([len(self._raw_input.shape)]),
                torch.tensor(self._raw_input.shape),
                torch.tensor([1]),
                self._raw_input.view(-1),
            )
        )

        self.output = torch.cat(
            (
                torch.tensor([len(self._raw_output.shape)]),
                torch.tensor(self._raw_output.shape),
                torch.tensor([1]),
                self._raw_output.view(-1),
            )
        )
