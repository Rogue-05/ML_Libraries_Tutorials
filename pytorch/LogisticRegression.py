import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


bc=datasets.load_breast_cancer()
X,Y=bc.data,bc.target
n_samples,n_features=X.shape

X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=1234)
sc=StandardScaler() #recommended for logistic regression
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
x=torch.from_numpy(X_train.astype(np.float32))
xt=torch.from_numpy(X_test.astype(np.float32))
y=torch.from_numpy(Y_train.astype(np.float32))
yt=torch.from_numpy(Y_test.astype(np.float32))

y=y.reshape(-1,1)
yt=yt.reshape(-1,1)

#model
class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)

    def forward(self,x):
        y_pred=torch.sigmoid(self.linear(x))
        return y_pred

model=LogisticRegression(n_features)

#loss and optimizer
loss=nn.BCELoss()
optimizer=optim.SGD(model.parameters(), lr=0.01)

#Training loop
epochs_num=100
for epoch in range(epochs_num):
    y_pred=model(x)
    l=loss(y_pred,y)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}: loss={l.item():.3f}')

with torch.no_grad():
    y_pred=model(xt)
    y_pred_cls=y_pred.round()
    acc=(y_pred_cls.eq(yt).sum())/float(yt.shape[0])
    print(f'Accuracy: {acc:.4f}')

# Visualizing the decision boundary
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap='bwr', alpha=0.5)
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap='bwr', alpha=0.5)
plt.show()