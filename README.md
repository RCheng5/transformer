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
Declare ODE using $$A = \begin{bmatrix} -2 & 1 \\ -4 & 1 \end{bmatrix}$$. Loads datapoints into tensor T.

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
Convert datapoints into batches for feeding. The number of dimensions is $3$ from $x, y, t$.
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
```
Y = X.detach().squeeze(0).numpy()
plt.figure()
plt.plot(T[:, 0], T[:, 1], '.')
plt.plot(Y[:, 0], Y[:, 1], 'r')
```
Outputs predicted curve relative to actual curve.
