import numpy as np

#f=w*x
#f=2*x

X=np.array([1,2,3,4],dtype=np.float32)
Y=np.array([2,4,6,8],dtype=np.float32)

w=0.0 

#model prediction
def forward(x):
    return w * x

#loss
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

#MSE=1/N *(w*x-y)^2
#dJ/dw=1/N * (w*x-y)*2*x

#gradient
def gradient(x, y, y_pred):
    return np.dot(2*(x),y_pred-y).mean()


print(f'Prediction before training: {forward(5):.3f}')

#training
learning_rate = 0.01
n_iters = 20
for epoch in range(n_iters):
    # prediction
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradient
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f'Epoch {epoch+1}: w={w:.3f}, loss={l:.3f}')

print(f'Prediction after training: {forward(5):.3f}')