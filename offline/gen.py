import torch
import torch.nn.functional as F
import random

from Datapoint import DataPoint
from Shape import Shape, KernelShape
from range import *


def tensor_gen(shape: tuple, mode="random", val=None):
    """
    Return a tensor with the given shape.
    """
    # uniform distribution in [-1, 1)
    if mode == "random":
        # return torch.rand(shape)
        # [0, 1) -> [-1, 1)
        return torch.rand(shape) * 2 - 1
    # normal distribution in [0, 1)
    elif mode == "normal":
        return torch.randn(shape)
    elif mode == "test":
        return torch.full(shape, val)
    else:
        assert False


def mul_gen(batch_size=1):
    dps = []
    
    dps = []
    for _ in range(batch_size):
        input_1 = tensor_gen(
            (1, input_shape.channel, input_shape.height, input_shape.width)
        )
        input_2 = tensor_gen(
            (1, input_shape.channel, input_shape.height, input_shape.width)
        )
        input = torch.cat((input_1, input_2), dim=0)
        output = torch.dot(input_1, input_2)
        dps.append(DataPoint(input, output, num_input=2))

    return dps
        
        
def argmax_gen(batch_size=1):
    dps = []
    
    for _ in range(batch_size):
        input = tensor_gen((1, 10))
        output = torch.argmax(input).float()
        dps.append(DataPoint(input, output))
    
    return dps
        
        
def concat_gen(batch_size=1):
    input_shape = Shape.get_random_shape()
    
    dps = []
    for _ in range(batch_size):
        input_1 = tensor_gen(
            (1, input_shape.channel, input_shape.height, input_shape.width)
        )
        input_2 = tensor_gen(
            (1, input_shape.channel, input_shape.height, input_shape.width)
        )
        input = torch.cat((input_1, input_2), dim=0)
        output = torch.cat((input_1, input_2), dim=0)
        dps.append(DataPoint(input, output, num_input=2, info={"input_shape": input_shape}))

    return dps


def add_gen(batch_size=1):
    input_shape = Shape.get_random_shape()

    dps = []
    for _ in range(batch_size):
        input_1 = tensor_gen(
            (1, input_shape.channel, input_shape.height, input_shape.width)
        )
        input_2 = tensor_gen(
            (1, input_shape.channel, input_shape.height, input_shape.width)
        )
        input = torch.cat((input_1, input_2), dim=0)
        output = input_1 + input_2
        dps.append(DataPoint(input, output, num_input=2, info={"input_shape": input_shape}))

    return dps


def batch_norm_gen(batch_size=1):
    """
    Deprecated
    Generate dp for BatchNorm2d
    """
    input_shape = Shape.get_random_shape()
    num_channel = input_shape.channel

    # mean and var are calculated per channel
    mean = tensor_gen((num_channel))
    var = tensor_gen((num_channel))

    dps = []
    for _ in range(batch_size):
        input = tensor_gen(
            (1, input_shape.channel, input_shape.height, input_shape.width)
        )
        output = F.batch_norm(input, mean, var)
        dps.append(DataPoint(input, output))

    return dps


def sigmoid_gen(batch_size=1):
    """
    Deprecated
    """
    input_shape = Shape.get_random_shape()

    dps = []
    for _ in range(batch_size):
        input = tensor_gen((input_shape.channel, input_shape.height, input_shape.width))
        output = torch.sigmoid(input)
        dps.append(DataPoint(input, output))

    return dps


def softmax_gen(batch_size=1):
    # assume only flat shape
    input_shape = Shape(channel=1, height=1, width=random.choice(shape_range))

    dps = []
    for _ in range(batch_size):
        input = tensor_gen((input_shape.channel, input_shape.height, input_shape.width))
        assert input.shape[0] == input.shape[1] == 1
        assert input.dim() == 3
        output = F.softmax(input, dim=2)
        dps.append(DataPoint(input, output, info={"input_shape": input_shape}))

    return dps


def relu_gen(batch_size=1):
    input_shape = Shape.get_random_shape()

    dps = []
    for _ in range(batch_size):
        input = tensor_gen((input_shape.channel, input_shape.height, input_shape.width))
        output = F.relu(input)
        dps.append(DataPoint(input, output, info={"input_shape": input_shape}))

    return dps


def prelu_gen(batch_size=1):
    input_shape = Shape.get_random_shape()

    dps = []
    for _ in range(batch_size):
        input = tensor_gen((input_shape.channel, input_shape.height, input_shape.width))
        output = F.prelu(input, random.random())
        dps.append(DataPoint(input, output))

    return dps


def elu_gen(batch_size=1):
    input_shape = Shape.get_random_shape()

    dps = []
    for _ in range(batch_size):
        input = tensor_gen((input_shape.channel, input_shape.height, input_shape.width))
        output = F.elu(input, random.random())
        dps.append(DataPoint(input, output))

    return dps


def flatten_gen(batch_size=1):
    input_shape = Shape.get_random_shape()

    dps = []
    for _ in range(batch_size):
        input = tensor_gen((input_shape.channel, input_shape.height, input_shape.width))
        output = torch.flatten(input)
        dps.append(DataPoint(input, output))

    return dps


def squeeze_gen(batch_size=1):
    input_shape = Shape.get_random_shape()

    dps = []
    for _ in range(batch_size):
        input = tensor_gen((input_shape.channel, input_shape.height, input_shape.width))
        # TODO: randomly insert some singleton dimension
        output = torch.squeeze(input)
        dps.append(DataPoint(input, output))

    return dps


def unsqueeze_gen(batch_size=1):
    input_shape = Shape.get_random_shape()

    dps = []
    for _ in range(batch_size):
        input = tensor_gen((input_shape.channel, input_shape.height, input_shape.width))
        dim = random.choices(range(input.dim() + 1))[0]
        output = torch.unsqueeze(input, dim)
        dps.append(DataPoint(input, output))

    return dps


def transpose_gen(batch_size=1):
    input_shape = Shape.get_random_shape()
    
    dps = []
    for _ in range(batch_size):
        input = tensor_gen((1, input_shape.channel, input_shape.height, input_shape.width))
        output = input.permute(0, 2, 3, 1)
        dps.append(DataPoint(input, output))

    return dps


def avg_pool_gen(batch_size=1):
    i_s = Shape.get_random_shape()
    k_s = KernelShape.get_random_kernel_shape()

    dps = []
    for _ in range(batch_size):
        input = tensor_gen((1, i_s.channel, i_s.height, i_s.height), mode="normal")
        output = F.avg_pool2d(
            input, (k_s.height, k_s.width), stride=k_s.stride, padding=k_s.pad
        )
        dps.append(
            DataPoint(input, output, info={"input_shape": i_s, "kernel_shape": k_s})
        )

    return dps


def max_pool_gen(batch_size=1):
    i_s = Shape.get_random_shape()
    k_s = KernelShape.get_random_kernel_shape()

    dps = []
    for _ in range(batch_size):
        input = tensor_gen((1, i_s.channel, i_s.height, i_s.height), mode="normal")
        output = F.max_pool2d(
            input, (k_s.height, k_s.width), stride=k_s.stride, padding=k_s.pad
        )
        dps.append(
            DataPoint(input, output, info={"input_shape": i_s, "kernel_shape": k_s})
        )

    return dps


def fc_gen(batch_size=1):
    """
    Generate dp for FC.
    ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html#torch.nn.functional.linear
    """
    num_input = random.choice(shape_range)
    num_hidden = random.choice(shape_range)
    num_output = random.choice(shape_range)

    input_shape = Shape(channel=None, height=num_input, width=num_hidden)
    weights_shape = Shape(channel=None, height=num_hidden, width=num_output)

    # be careful here, weights: (out, in), height/width is confusing here
    weights = tensor_gen((weights_shape.width, weights_shape.height))
    bias = tensor_gen((weights_shape.width))

    dps = []
    for _ in range(batch_size):
        input = tensor_gen((input_shape.height, input_shape.width), mode="normal")
        output = F.linear(input, weights, bias)
        dps.append(DataPoint(input, output, weight_0=weights, weight_1=bias, info={"input_shape": input_shape, "weights_shape": weights_shape}))

    return dps


def conv_gen(batch_size=1):
    input_shape = Shape.get_random_shape()
    kernel_shape = KernelShape.get_random_kernel_shape()

    # randomize kernel weights and bias
    assert kernel_shape.group == 1
    # filter_size = i_s.channel * k_s.output_channel // k_s.group
    weights = tensor_gen(
        (
            kernel_shape.output_channel,
            input_shape.channel,
            kernel_shape.height,
            kernel_shape.width,
        )
    )
    bias = tensor_gen(kernel_shape.output_channel)

    dps = []
    for _ in range(batch_size):
        input = tensor_gen(
            (1, input_shape.channel, input_shape.height, input_shape.width),
            mode="normal",
        )
        # conv
        output = F.conv2d(
            input,
            weights,
            bias=bias,
            stride=kernel_shape.stride,
            padding=kernel_shape.pad,
            dilation=kernel_shape.dilation,
            groups=kernel_shape.group,
        )
        dps.append(DataPoint(input, output, weight_0=weights, weight_1=bias, info={"input_shape": input_shape, "kernel_shape": kernel_shape}))

    return dps


def conv_relu_gen(batch_size=1):
    input_shape = Shape.get_random_shape()
    kernel_shape = KernelShape.get_random_kernel_shape()

    # randomize kernel weights and bias
    assert kernel_shape.group == 1
    # filter_size = i_s.channel * k_s.output_channel // k_s.group
    weights = tensor_gen(
        (
            kernel_shape.output_channel,
            input_shape.channel,
            kernel_shape.height,
            kernel_shape.width,
        )
    )
    bias = tensor_gen(kernel_shape.output_channel)

    dps = []
    for _ in range(batch_size):
        input = tensor_gen(
            (1, input_shape.channel, input_shape.height, input_shape.width),
            mode="normal",
        )
        # conv
        output = F.conv2d(
            input,
            weights,
            bias=bias,
            stride=kernel_shape.stride,
            padding=kernel_shape.pad,
            dilation=kernel_shape.dilation,
            groups=kernel_shape.group,
        )
        # relu
        output = F.relu(output)
        dps.append(DataPoint(input, output, weight_0=weights, weight_1=bias, info={"input_shape": input_shape, "kernel_shape": kernel_shape}))

    return dps


def conv_bn_gen(batch_size=1):
    """
    Deprecated 
    """
    input_shape = Shape.get_random_shape()
    kernel_shape = KernelShape.get_random_kernel_shape()

    # randomize kernel weights and bias
    assert kernel_shape.group == 1
    # filter_size = i_s.channel * k_s.output_channel // k_s.group
    weights = tensor_gen(
        (
            kernel_shape.output_channel,
            input_shape.channel,
            kernel_shape.height,
            kernel_shape.width,
        )
    )
    bias = tensor_gen(kernel_shape.output_channel)

    # mean and var are calculated per output_channel
    mean = tensor_gen((kernel_shape.output_channel))
    var = tensor_gen((kernel_shape.output_channel))

    dps = []
    for _ in range(batch_size):
        input = tensor_gen(
            (1, input_shape.channel, input_shape.height, input_shape.width),
            mode="normal",
        )
        # conv
        output = F.conv2d(
            input,
            weights,
            bias=bias,
            stride=kernel_shape.stride,
            padding=kernel_shape.pad,
            dilation=kernel_shape.dilation,
            groups=kernel_shape.group,
        )
        # bn
        output = F.batch_norm(output, mean, var)
        dps.append(DataPoint(input, output))

    return dps
