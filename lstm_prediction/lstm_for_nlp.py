import torch
import torch.nn.functional as F
from pandas import read_csv
from pandas import datetime
from numpy.random import seed

seed(1)


train_set = read_csv('dataset/ShangHai_comfirmed.csv', header=None)
train_value = train_set.values
test_tensor = torch.Tensor([128])

class One_layer_perception(torch.nn.Module) :
    def __init__(self):
        super(One_layer_perception, self).__init__()
        self.Linear_layer = torch.nn.Linear(1,32,bias=False)

    def forward(self, trains_set):
        y_hat = F.relu(self.Linear_layer(trains_set))
        return y_hat

model = One_layer_perception()

print(model(test_tensor))