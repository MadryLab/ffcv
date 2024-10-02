"""
Example of using FFCV to speed up large scale linear regression.
For tutorial, see https://docs.ffcv.io/ffcv_examples/linear_regression.html.

"""
from tqdm import tqdm
import time
import numpy as np
import pickle as pkl
import torch as ch
from torch.utils.data import TensorDataset, DataLoader
from ffcv.fields import NDArrayField, FloatField
from ffcv.fields.basics import FloatDecoder
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter
from ffcv.transforms import ToTensor, ToDevice, Squeeze
import os

# 1,000,000 inputs each of dimension 10,000 = 40GB of data
N, D = 1000000, 10000
USE_FFCV = True
if not os.path.exists('/tmp/linreg_data.pkl'):
    X = np.random.rand(N, D).astype('float32')
    # Ground-truth vector
    W, b = np.random.rand(D).astype('float32'), np.random.rand()
    # Response variables
    Y = X @ W + b + np.random.randn(N).astype('float32')
    pkl.dump((X, W, b, Y), open('/tmp/linreg_data.pkl', 'wb'))
elif not USE_FFCV:
    print('Loading from disk...')
    X, W, b, Y = pkl.load(open('/tmp/linreg_data.pkl', 'rb'))

if USE_FFCV and not os.path.exists('/tmp/linreg_data.beton'):
    X, W, b, Y = pkl.load(open('/tmp/linreg_data.pkl', 'rb'))
    class LinearRegressionDataset:
        def __getitem__(self, idx):
            return (X[idx], np.array(Y[idx]).astype('float32'))

        def __len__(self):
            return len(X)

    writer = DatasetWriter('/tmp/linreg_data.beton', {
        'covariate': NDArrayField(shape=(D,), dtype=np.dtype('float32')),
        'label': NDArrayField(shape=(1,), dtype=np.dtype('float32')),
    }, num_workers=16)

    writer.from_indexed_dataset(LinearRegressionDataset())
else:
    print('FFCV file already written')


### PART 2: actual regression

if not USE_FFCV:
    dataset = TensorDataset(ch.tensor(X), ch.tensor(Y))
    train_loader = DataLoader(dataset, batch_size=2048, num_workers=8, shuffle=True)
else:
    train_loader = Loader('/tmp/linreg_data.beton', batch_size=2048,
                    num_workers=8, order=OrderOption.QUASI_RANDOM, cache_type=1,
                    pipelines={
                        'covariate': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'))],
                        'label': [NDArrayDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'))]
                    })

# Calculate data mean and variance for normalization
def calculate_stats(loader, N):
    mean, stdev = 0., 0.
    for x_batch, _ in tqdm(loader):
        mean += x_batch.sum(0) / N
        stdev += x_batch.pow(2).sum(0) / N
    return mean, ch.sqrt(stdev - mean.pow(2))

mean, stdev = calculate_stats(train_loader, N)
mean, stdev = mean.cuda(), stdev.cuda()
w_est, b_est = ch.zeros(D).cuda(), ch.zeros(1).cuda() # Initial guess for W
num_epochs = 10 # Number of full passes over the data to do

lr = 5e-2
for _ in range(num_epochs):
    total_loss, num_examples = 0., 0.
    start_time = time.time()
    for (x_batch, y_batch) in tqdm(train_loader):
        if not USE_FFCV:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        # Normalize the data for stability
        x_batch = (x_batch - mean) / stdev
        residual = x_batch @ w_est + b_est - y_batch
        # Gradients
        w_grad = x_batch.T @ residual / x_batch.shape[0]
        b_grad = ch.mean(residual, dim=0)
        w_est = w_est - lr * w_grad
        b_est = b_est - lr * b_grad
        total_loss += residual.pow(2).sum()
        num_examples += x_batch.shape[0]
    print('Epoch time:', time.time() - start_time)
    print(f'Average loss: {total_loss / num_examples:.3f}')
