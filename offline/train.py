import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split

from Dataset import get_dp, Operator, CustomDataset, collate_fn
from Model import FeatureNet, HybridNet
from utils import load_dataset

import log
import logging
logger = log.setup_custom_logger(name='log', level='DEBUG')

def calculate_accuracy_and_confusion_matrix(loader, model, num_classes, device):
    model.eval()  # Set the model to evaluation mode
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for (padded_input_sequence_data, input_lens), (padded_output_sequence_data, output_lens), feature_data, labels in loader:
            feature_data = feature_data.to(device)
            labels = labels.to(device)
            
            outputs = model(padded_input_sequence_data, input_lens, padded_output_sequence_data, output_lens, feature_data)        
            _, predicted = torch.max(outputs, 1)
            for label, prediction in zip(labels, predicted):
                confusion_matrix[label.long(), prediction.long()] += 1

    # Calculating accuracy for each class
    for i in range(num_classes):
        correct = confusion_matrix[i, i].item()
        total = confusion_matrix[i].sum().item()
        if total > 0:
            accuracy = 100 * correct / total
            logger.info(f'Accuracy of class {i}: {accuracy:.2f}%')
        else:
            logger.info(f'Class {i} has no samples')
    
    # Printing confusion matrix
    logger.info('Confusion Matrix:')
    logger.info(confusion_matrix)
    return confusion_matrix


def calculate_accuracy_per_class(loader, model, num_classes, device):
    model.eval()  # Set the model to evaluation mode
    class_correct = [0. for i in range(num_classes)]
    class_total = [0. for i in range(num_classes)]
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for (padded_input_sequence_data, input_lens), (padded_output_sequence_data, output_lens), feature_data, labels in loader:

            feature_data = feature_data.to(device)
            labels = labels.to(device)
            
            outputs = model(padded_input_sequence_data, input_lens, padded_output_sequence_data, output_lens, feature_data)        
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                total_correct += c[i].item()
                total_samples += 1

    overall_accuracy = 100 * total_correct / total_samples
    logger.info(f'Overall Accuracy: {overall_accuracy:.2f}%')
    for i in range(num_classes):
        if class_total[i] > 0:
            logger.info(f'Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%')
        else:
            logger.info(f'Accuracy of class {i}: N/A (no samples)')
            

def classification_training(dataset, device, epochs = 10):
    num_label=len(list(Operator))
    
    # dataset partitioning
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoader for classification 
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize the network
    ftr_len = train_dataset[0][0].features_len()
    print("feature size", ftr_len)
    model = FeatureNet(ftr_len=ftr_len, num_label=num_label).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  # Change this based on your task (e.g., nn.CrossEntropyLoss for classification)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # confusion matrix
    train_confusion_matrix_list = []
    test_confusion_matrix_list = []

    # Training Loop
    for epoch in range(epochs):
        model.train()  # Ensure the model is in training mode
        for batch_idx, ((padded_input_sequence_data, input_lens), (padded_output_sequence_data, output_lens), feature_data, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            feature_data = feature_data.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(padded_input_sequence_data, input_lens, padded_output_sequence_data, output_lens, feature_data)        
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

        # Calculate accuracy on training and test sets
        logger.info(f'Epoch {epoch + 1} Training Accuracy per Class:')
        train_confusion_matrix_list.append(calculate_accuracy_and_confusion_matrix(train_loader, model, num_label, device))
        logger.info(f'Epoch {epoch + 1} Test Accuracy per Class:')
        test_confusion_matrix_list.append(calculate_accuracy_and_confusion_matrix(test_loader, model, num_label, device))
        logger.info("-------------------------------------------")

    logger.info("Training Complete")
    
    return model, train_confusion_matrix_list, test_confusion_matrix_list


def calculate_regression_accuracy(loader, model, device):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for ((padded_input_sequence_data, input_lens), (padded_output_sequence_data, output_lens), feature_data, labels) in loader:
            padded_input_sequence_data = padded_input_sequence_data.to(device)
            padded_output_sequence_data = padded_output_sequence_data.to(device)
            feature_data = feature_data.to(device)
            labels = labels.to(device)
            
            outputs = model(padded_input_sequence_data, input_lens, padded_output_sequence_data, output_lens, feature_data)            
            total_correct += (outputs.squeeze().round().int() == labels).sum().item()
            total_count += len(outputs)
            
    overall_accuracy = 100 * total_correct / total_count
    logger.info(f'Overall Accuracy: {overall_accuracy:.2f}%')
    return overall_accuracy


def regression_training(dataset, device, epochs=100):
    '''
    Assume the labels have been adjusted for the task
    '''
    
    # dataset partitioning
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print("train_size: ", train_size, " test_size: ", test_size)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize the network
    ftr_len = train_dataset[0][0].features_len()
    model = FeatureNet(ftr_len=ftr_len, num_label=1).to(device) # regressional model
    # model = HybridNet(ftr_len=ftr_len).to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_acc_list = []
    test_acc_list = []

    # Training Loop
    for epoch in range(epochs):
        model.train()  # Ensure the model is in training mode
        for batch_idx, ((padded_input_sequence_data, input_lens), (padded_output_sequence_data, output_lens), feature_data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            padded_input_sequence_data = padded_input_sequence_data.to(device)
            padded_output_sequence_data = padded_output_sequence_data.to(device)
            feature_data = feature_data.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(padded_input_sequence_data, input_lens, padded_output_sequence_data, output_lens, feature_data)            
            loss = criterion(outputs, labels.unsqueeze(1).float())

            # Backward and optimize
            loss.backward()
            optimizer.step()

        # Calculate accuracy on training and test sets
        logger.info(f'Epoch {epoch + 1} Training Accuracy:')
        train_acc_list.append(calculate_regression_accuracy(train_loader, model, device))
        logger.info(f'Epoch {epoch + 1} Test Accuracy:')
        test_acc_list.append(calculate_regression_accuracy(test_loader, model, device))
        logger.info("-------------------------------------------")

    logger.info("Training Complete")
    best_train_acc = max(train_acc_list)
    best_test_acc = max(test_acc_list)
    logger.info(f"Best training acc: {best_train_acc}, best testing acc: {best_test_acc}")
    
    return model


def train_conv_kernel_size(dataset, device):
    op_name = [op.name for op in list(Operator)]
    
    # construct kernel size dataset from general dataset
    conv_dps = []
    conv_kernel_size = []
    for dp, label in dataset:
        if op_name[label] == "CONV" or op_name[label] == "CONV_RELU":
            conv_dps.append(dp)
            conv_kernel_size.append(dp.info['kernel_shape'].width)

    conv_kernel_size_dataset = CustomDataset(conv_dps, conv_kernel_size)

    model = regression_training(conv_kernel_size_dataset, device, epochs=200)
    
    return model


def train_conv_kernel_num(dataset, device):
    op_name = [op.name for op in list(Operator)]
    
    conv_dps = []
    conv_output_channel = []
    for dp, label in dataset:
        if op_name[label] == "CONV" or op_name[label] == "CONV_RELU":
            conv_dps.append(dp)
            conv_output_channel.append(dp.info['kernel_shape'].output_channel)

    conv_output_channel_dataset = CustomDataset(conv_dps, conv_output_channel)

    model = regression_training(conv_output_channel_dataset, device, epochs=200)
    
    return model


def train_conv_stride(dataset, device):
    op_name = [op.name for op in list(Operator)]
    
    conv_dps = []
    conv_stride = []
    for dp, label in dataset:
        if op_name[label] == "CONV" or op_name[label] == "CONV_RELU":
            conv_dps.append(dp)
            conv_stride.append(dp.info['kernel_shape'].stride)

    conv_stride_dataset = CustomDataset(conv_dps, conv_stride)

    model = regression_training(conv_stride_dataset, device, epochs=200)
    
    return model


def train_fc_output_size(dataset, device):
    op_name = [op.name for op in list(Operator)]

    fc_dps = []
    fc_output_size = []
    for dp, label in dataset:
        if op_name[label] == 'FC':
            fc_dps.append(dp)
            fc_output_size.append(dp.info['weights_shape'].width)

    fc_output_size_dataset = CustomDataset(fc_dps, fc_output_size)

    model = regression_training(fc_output_size_dataset, device, epochs=200)
    
    return model


def train_avgpool_kernel_size(dataset, device):
    op_name = [op.name for op in list(Operator)]

    ap_dps = []
    ap_kernel_size = []
    for dp, label in dataset:
        if op_name[label] == 'AVG_POOL':
            ap_dps.append(dp)
            ap_kernel_size.append(dp.info['kernel_shape'].width)

    ap_kernel_size_dataset = CustomDataset(ap_dps, ap_kernel_size)

    model = regression_training(ap_kernel_size_dataset, device, epochs=200)
    
    return model


def train_avgpool_padding_size(dataset, device):
    op_name = [op.name for op in list(Operator)]

    ap_dps = []
    ap_pad_size = []
    for dp, label in dataset:
        if op_name[label] == 'AVG_POOL':
            ap_dps.append(dp)
            ap_pad_size.append(dp.info['kernel_shape'].pad)

    ap_pad_size_dataset = CustomDataset(ap_dps, ap_pad_size)

    model = regression_training(ap_pad_size_dataset, device, epochs=200)
    
    return model


def train_avgpool_stride_size(dataset, device):
    op_name = [op.name for op in list(Operator)]

    ap_dps = []
    ap_stride_size = []
    for dp, label in dataset:
        if op_name[label] == 'AVG_POOL':
            ap_dps.append(dp)
            ap_stride_size.append(dp.info['kernel_shape'].stride)

    ap_stride_size_dataset = CustomDataset(ap_dps, ap_stride_size)

    model = regression_training(ap_stride_size_dataset, device, epochs=200)
    
    return model


def train_maxpool_kernel_size(dataset, device):
    op_name = [op.name for op in list(Operator)]

    mp_dps = []
    mp_kernel_size = []
    for dp, label in dataset:
        if op_name[label] == 'MAX_POOL':
            mp_dps.append(dp)
            mp_kernel_size.append(dp.info['kernel_shape'].width)

    mp_kernel_size_dataset = CustomDataset(mp_dps, mp_kernel_size)

    model = regression_training(mp_kernel_size_dataset, device, epochs=200)
    
    return model


def train_maxpool_padding_size(dataset, device):
    op_name = [op.name for op in list(Operator)]

    mp_dps = []
    mp_padding_size = []
    for dp, label in dataset:
        if op_name[label] == 'MAX_POOL':
            mp_dps.append(dp)
            mp_padding_size.append(dp.info['kernel_shape'].pad)

    mp_padding_size_dataset = CustomDataset(mp_dps, mp_padding_size)

    model = regression_training(mp_padding_size_dataset, device, epochs=200)
    
    return model


def train_maxpool_stride_size(dataset, device):
    op_name = [op.name for op in list(Operator)]

    mp_dps = []
    mp_stride_size = []
    for dp, label in dataset:
        if op_name[label] == 'MAX_POOL':
            mp_dps.append(dp)
            mp_stride_size.append(dp.info['kernel_shape'].stride)

    mp_stride_size_dataset = CustomDataset(mp_dps, mp_stride_size)

    model = regression_training(mp_stride_size_dataset, device, epochs=200)
    
    return model

    
def main():    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    PATH = "./saved_models"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    dataset = load_dataset("./dataset")
    for dp, label in dataset:
        dp.recompute()
    
    # train M_{t}
    mt_model, _, _ = classification_training(dataset, device, epochs=10)
    torch.save(mt_model.state_dict(), PATH + "/mt_model.pth")
    
    # train M_{att}
    
    # conv 
    model = train_conv_kernel_size(dataset, device)
    torch.save(model.state_dict(), PATH + "/matt_conv_kernel_size.pth")
    
    model = train_conv_kernel_num(dataset, device)
    torch.save(model.state_dict(), PATH + "/matt_conv_kernel_num.pth")
    
    model = train_conv_stride(dataset, device)
    torch.save(model.state_dict(), PATH + "/matt_conv_kernel_stride.pth")
    
    # fc 
    model = train_fc_output_size(dataset, device)
    torch.save(model.state_dict(), PATH + "/matt_fc_output_size.pth")
    
    # avgpool
    model = train_avgpool_kernel_size(dataset, device)
    torch.save(model.state_dict(), PATH + "/matt_avgpool_kernel_size.pth")
    
    model = train_avgpool_padding_size(dataset, device)
    torch.save(model.state_dict(), PATH + "/matt_avgpool_padding_size.pth")
    
    model = train_avgpool_stride_size(dataset, device)
    torch.save(model.state_dict(), PATH + "/matt_avgpool_stride_size.pth")
    
    # maxpool
    model = train_maxpool_kernel_size(dataset, device)
    torch.save(model.state_dict(), PATH + "/matt_maxpool_kernel_size.pth")
    
    model = train_maxpool_padding_size(dataset, device)
    torch.save(model.state_dict(), PATH + "/matt_maxpool_padding_size.pth")
    
    model = train_maxpool_stride_size(dataset, device)
    torch.save(model.state_dict(), PATH + "/matt_maxpool_stride_size.pth")
    
        
if __name__ == "__main__":
    main()