#design model (input,output,forward pass)
#construct loss and optimizer
#Training loop:
   #forward pass -> backward pass -> update weights

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X_numpy,Y_numpy=datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
x=torch.from_numpy(X_numpy.astype(np.float32))
y=torch.from_numpy(Y_numpy.astype(np.float32))
y=y.reshape(-1,1)
n_samples, n_features = x.shape
#1) Model 

model=nn.Linear(n_features,1)

#2) Loss and optimizer
loss=nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
#3) Training loop
epochs_num=100
for epoch in range(epochs_num):
    y_pred=model(x)
    l=loss(y,y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        [w,b]=model.parameters()
        print(f'Epoch {epoch+1}: w={w[0].item():.3f}, b={b[0].item():.3f}, loss={l:.3f}')

predi=model(x).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro', label='Original data')
plt.plot(X_numpy,predi,'b')
plt.show()