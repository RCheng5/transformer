# Transformer
The goal is to train a transformer to model the ODE $x' = Ax$. The transformer is trained on some datapoints and it should be able to output later datapoints.
```
import os
from time import time

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim

from PIL import Image
from torchvision import transforms

x = torch.rand(5, 3);
y = torch.ones_like(x, dtype=torch.float)
print("x = ", x);
print("y = ", y);
print("x stored in ", x.device);

print("cuda = ", torch.cuda.is_available());
print("ver = ", torch.__version__);
```
import libraries
```
# ODE, x' = Ax
A = [[-2, 1], [-4, 1]]
A = np.array(A) 

lam, X = np.linalg.eig(A)
print('lambda =', lam)
print('X =', X)
v1 = X[:,0]
v2 = X[:,1]

print(v1, A@v1, lam[0] * v1)
print(v2, A@v2, lam[1] * v2)

# v(t) = v1*exp(lambda1 * t) + v2*exp(lambda2 * t)
dt = 0.1
t = np.arange(0, 6, dt)
num = len(t)
T = np.zeros([3, num])
for i in range(num):
    T[:2, i] = v1 * np.exp(lam[0]*t[i]) + v2 * np.exp(lam[1]*t[i])

T[2,:] = t
T = np.real(T)

plt.figure()
plt.plot(T[0, :], T[1, :], '.')

T = T.transpose()
T = torch.tensor(T).float()
print(T.shape)
```
Declare ODE $x' = Ax$ using 
```math
A = \begin{bmatrix} -2 & 1 \\ -4 & 1 \end{bmatrix}
```
The initial condition is at $t=0$ and 
```math
\begin{bmatrix}0.67082039 \\ 1.78885438 \end{bmatrix}
```
This is just the sum of the eigenvectors of $A$, $v_1$ and $v_2$.
To find the curve described by this equation, I used python to find the 2 eigenvectors $v_1$ and $v_2$ and their respective eigenvalues of $\lambda_1$ and $\lambda_2$. Then the solution is described by the equation $$x(t) = Cv_1e^{\lambda_1 t} + Dv_2e^{\lambda_2 t}$$ where $C$ and $D$ are arbitrary constants. For the curve I generated, I used $C=D=1$ so the curve follows the equation $x(t) = v_1e^{\lambda_1 t} + v_2e^{\lambda_2 t}$ where $0 \leq t < 6$. I generated data points on the curve at intervals of $dt = 0.1$. This code generates the data points and plots them. Each datapoint is a tensor of size 3, with the 1st dimension being x-value, 2nd dimension being y-value, and 3rd dimension being the time t. The datapoints are loaded into a tensor T with size $60\times3$ where $60$ is the total number of datapoints generated.

```
tf = torch.nn.Transformer(d_model=3, nhead=1, 
                          num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64, dropout=0.1,
                          batch_first=True, norm_first=False, device=None, dtype=None)

print(tf)
```
Declare transformer
```
# convert double tensor to float and add batch dimension
x = T[:6,:].unsqueeze(0)
y = torch.zeros(1, 3).unsqueeze(0)
y[0,0,2] = x[0, -1, 2]
print('x =', x, x.size(0), x.size(1), x.size(2))
print('y =', y)
z = tf(x, y)
print('z =', z)

# load a sequence, the next prediction and the next hint from S
def get_seq(S, pos, size = 8):
    x = S[pos:(pos+size), :].unsqueeze(0)
    y = S[(pos + size):(pos + size + 1), :].unsqueeze(0)
    h = y.clone()
    h[0, 0, :2] = 0
    return x, y, h

x, y, h = get_seq(T, 0)
print('x =', x)
print('y =', y)
print('h =', h)

z = tf(x, h)
print('z =', z)

print(x[0, :-1, :])
```
Convert datapoints into batches for feeding. get_seq() is the function for generating batches. S is the tensor of datapoints passed in, in this case S will be T. pos is the beginning datapoint of the batch. size is the size of the batch. It returns x, y, h where x is a tensor of size $1\times 8 \times 3$ consisting of the datapoints between pos and pos + size - 1. y is a tensor of size $1 \times 1 \times 3$ consisting of the datapoint at pos + size. h is a tensor of size $1\times 1\times 3$ where each variable is 0. x and h will be passed into the model for training.
```
# train
S = T[:30, :]

optimizer = optim.SGD(tf.parameters(), lr = 0.003, momentum = 0.9)
criterion = nn.L1Loss()
seq = 8

time0 = time()
num = S.size(0) - seq;

for epoch in range(400):
    rloss = 0
    lst = torch.randperm(num)
    for pos in lst:
        x, y, h = get_seq(S, pos, size = seq)
        optimizer.zero_grad();
        yp = tf(x, h)
        
        # -- learning classification
        loss = criterion(y, yp) # calculate the NLL loss
        loss.backward()
        optimizer.step()
        rloss += loss.item()
        
    if epoch % 20 == 0:
        print("Epoch {} - Training loss: {}"
          .format(epoch, rloss/num))
    
time1 = time()
print('time elapsed', time1 - time0, 'sec')
```
Trains the model
The tensor S is a tensor of size $30\times 3$ and consists of the first 30 datapoints in T. These will be fed into the transformer for training. Stochastic gradient descent (SGD) was the optimizer used and the loss function is N1loss. seq is the number of datapoints in each batch. In this case, seq is 8. The number of epochs is 400. During each epoch, lst is generated which is a random permutation of the numbers from 0 to S.size(0)-seq, that is a permutation of all the possible starting positions of each batch. A batch is created from get_seq(S, pos, size=seq) from each pos in lst. x and h from each batch are fed into the transformer to produce yp, a tensor of size $1\times 1 \times 3$ that is the predicted value of y from x. The transformer is optimized comparing y and yp.
```
# prediction
X, y, h = get_seq(S, 0)
x = X.clone()

t0 = h[0, 0, 2]
print('t0 =', t0)

task = [1]
if 1 in task:
    while t0 < 6 * 1.1:
        z = tf(x, h)
        X = torch.cat((X, z), 1)
        x[0, :-1, :] = x[0, 1:, :]
        x[0, -1, :] = z
        t0 += dt
        h[0, 0, 2] = t0

if 2 in task:
    h = torch.zeros(1, 60, 3)
    for i in range(60):
        h[0, i, 2] = t0
        t0 += dt
    z = tf(x, h)
    X = torch.cat((X, z), 1)

print(X.shape)
```
Prediction
tf is the transformer model. The first 8 datapoints are loaded into x. h is a tensor of the indicating the time of the next point. The transformer outputs a tensor z from z = tf(x, h) and z is the predicted next datapoint on the curve. 

For example, in the first iteration, the known datapoints from $t = 0, 0.1, 0.2, ..., 0.7$ are loaded in to the transformer and h is fed in where h is (0, 0, 0.8). The x tensor fed in looks like this, 
x = tensor([[[0.6708, 1.7889, 0.0000],
         [0.7068, 1.6867, 0.1000],
         [0.7258, 1.5623, 0.2000],
         [0.7293, 1.4200, 0.3000],
         [0.7186, 1.2643, 0.4000],
         [0.6953, 1.0994, 0.5000],
         [0.6609, 0.9292, 0.6000],
         [0.6173, 0.7576, 0.7000]]])
and h =  tensor([[[0.0000, 0.0000, 0.8000]]])
From this, z is calculated where 
z = tensor([[[0.3130, 0.1434, 0.8892]]]
z is the next datapoint.

After each iteration, a new batch of 8 is created with the last 7 points in the original batch being the first 7 in the new batch and with the calculated z being the last point in the batch. For example, in the second iteration,
x =  tensor([[[0.7068, 1.6867, 0.1000],
         [0.7258, 1.5623, 0.2000],
         [0.7293, 1.4200, 0.3000],
         [0.7186, 1.2643, 0.4000],
         [0.6953, 1.0994, 0.5000],
         [0.6609, 0.9292, 0.6000],
         [0.6173, 0.7576, 0.7000],
         [0.3130, 0.1434, 0.8892]]])
h will just indicate the time where the next datapoint should be calculated. For the second iteration, 
h =  tensor([[[0.0000, 0.0000, 0.9000]]])
Then another z is calculated and we continue to iterate until we have gone through all the datapoints.
All the z's will be stored in X so that we can created a predicted curve from the predicted points.
```
Y = X.detach().squeeze(0).numpy()
plt.figure()
plt.plot(T[:, 0], T[:, 1], '.')
plt.plot(Y[:, 0], Y[:, 1], 'r')
```
Outputs predicted curve (red) relative to actual curve (blue).
