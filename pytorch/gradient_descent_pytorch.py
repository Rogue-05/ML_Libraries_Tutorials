#design model (input,output,forward pass)
#construct loss and optimizer
#Training loop:
   #forward pass -> backward pass -> update weights

import torch
import torch.nn as nn
import torch.optim as optim

#f=w*x
#f=2*x

X=torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y=torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)
n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

#model prediction
model=nn.Linear(input_size, output_size)
#MSE=1/N *(w*x-y)^2
#dJ/dw=1/N * (w*x-y)*2*x

#gradient



print(f'Prediction before training: {model(torch.Tensor([5]))}')

#training
learning_rate = 0.01
n_iters = 100

loss=nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(n_iters):
    # prediction
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradient
    l.backward()

    # update weights
    optimizer.step()
    # zero the gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w,b]=model.parameters()
        print(f'Epoch {epoch+1}: w={w[0].item():.3f}, b={b[0].item():.3f}, loss={l:.3f}')

print(f'Prediction before training: {model(torch.Tensor([5])).item():.3f}')