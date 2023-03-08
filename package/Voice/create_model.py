import torch
from torch import nn
import torch.nn.functional as F

class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()
        # input 應該是 (10, 1, 40, 100)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) 
        self.cnn2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1) 
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) 

        self.cnn3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU() # activation
        self.cnn4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU() # activation
        self.cnn5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU() # activation

        self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride = 2) 

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(2048, 256) 
        self.relu4 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 128) 
        self.relu5 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(128, 30) 
        self.soft = nn.Softmax(dim = 1)
    
    def forward(self, x):
        out = self.cnn1(x)
        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.maxpool2(out)
        
        out = self.cnn3(out)
        out = self.relu1(out)

        out = self.cnn4(out)
        out = self.relu2(out)

        out = self.cnn5(out)
        out = self.relu3(out)

        out = self.maxpool3(out)

        out = self.flat(out)

        out = self.fc1(out)
        out = self.relu4(out)

        out = self.drop1(out)
        out = self.fc2(out)
        out = self.relu5(out)

        out = self.drop2(out)
        out = self.fc3(out)

        out = self.soft(out)
        return out

   
