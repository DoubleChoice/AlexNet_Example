import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        # output =
        self.a = nn.Sequential(
            nn.Conv2d(3,48,kernel_size=11,stride=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(48,128,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(128,192,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(192,192,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(192,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )

        self.b = nn.Sequential(
            nn.Linear(128*6*6,2048),
            nn.ReLU(),
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Linear(2048,5)
        )

    def forward(self,x):
        x = self.a(x)
        x= torch.flatten(x,start_dim=1)
        x  =self.b(x)
        return x

