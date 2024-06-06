import random
from range import *

class Shape():
    """
    Shape of input/output    
    """

    def __init__(self, channel, height, width):
        self.channel = channel
        self.height = height
        self.width = width
    
    @classmethod
    def get_random_shape(self):
        shape_len = random.choice(shape_range)
        num_channel = random.choice(channel_range)
        return Shape(channel=num_channel, 
                        height=shape_len,
                        width=shape_len)


class KernelShape(Shape):
    """
    Shape of Conv kernel
    """

    def __init__(self,
                 output_channel,
                 height,
                 width,
                 pad=(0, 0),
                 stride=(1, 1),
                 dilation=(1, 1),
                 group=1):
        super().__init__(None, height, width)
        self.output_channel = output_channel
        self.pad = pad
        self.stride = stride

        # not supported yet
        self.dilation = dilation
        self.group = group
    
    @classmethod
    def get_random_kernel_shape(self):
        shape_len = random.choice(kernel_shape_range)
        num_output_channel = random.choice(channel_range)
        # pad should be at most half of kernel size
        pad = random.choice([x for x in pad_range if x < shape_len/2])
        stride = random.choice(stride_range)
        return KernelShape(output_channel=num_output_channel,
                           height=shape_len,
                           width=shape_len,
                           pad=pad,
                           stride=stride)