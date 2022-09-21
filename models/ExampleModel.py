import torchmetrics
import torch
import torch.nn as nn
from .ModelBase import ModelBase

class SimpleConv(ModelBase):
    def __init__(
        self,
        metrics: torchmetrics.MetricCollection = torchmetrics.MetricCollection({}),
        learning_rate: float = 0.0001,
        num_classes: int = 2
    ):
        super().__init__(metrics=metrics)
        
        self.loss = nn.CrossEntropyLoss() 
        self.lr = learning_rate

        # Build model
        self.model = nn.Sequential()
        self.model.append(nn.Linear(128, 256)) #frame size after conv layers = 22
        self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Dropout())
        self.model.append(nn.Linear(256, 128))
        self.model.append(nn.ReLU(inplace=True))
        self.model.append(nn.Dropout())
        self.model.append(nn.Linear(128, num_classes))
        
    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.00001)