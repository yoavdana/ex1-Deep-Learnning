import torch
import numpy as np
import scipy.stats as si
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from data_proc import *
from net import *

INPUT_DIM = 9 * 20
OUTPUT_DIM = 2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def train(model, iterator_x, iterator_y, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for i, x in enumerate(iterator_x):
        x = x.to(device)
        y = iterator_y[i].to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator_x), epoch_acc / len(iterator_x)


if __name__ == "__main__":
    filename_pos='resorces/pos_A0201.txt'
    filename_neg='resorces/neg_A0201.txt'

    train_set, test_set=DATA_pre_pros(filename_pos,filename_neg)



    model = NetWork(INPUT_DIM, OUTPUT_DIM)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)


    BATCH_SIZE = 64
   # batch_x=train_set_x[0:BATCH_SIZE]
    #batch_y=train_set_y[0:BATCH_SIZE]
    #batch_x = torch.from_numpy(batch_x.astype('float32'))
    #batch_y = torch.from_numpy(batch_y.astype('float32'))
    #model.train()
    #y_p=model(batch_x)
    #print(y_p)


    EPOCHS = 10

    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):

        train_x,train_y=shuffle_data(train_set)
        train_iterator_x = torch.from_numpy(train_x.astype('float32')).split(64)
        train_iterator_y = torch.from_numpy(train_y.astype('int64')).split(64)

        train_loss, train_acc = train(model, train_iterator_x, train_iterator_y, optimizer, criterion, device)

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

