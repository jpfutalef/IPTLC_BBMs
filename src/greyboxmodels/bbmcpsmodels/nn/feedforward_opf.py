import torch.nn as nn
import torch.nn.functional as F


class BBM1_SimpleNet(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.15):
        super(BBM1_SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, output_size)
        self.dropout = nn.Dropout(p=dropout)

        # Get the name of the class
        self.name = self.__class__.__name__

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BBM2_DeepNN(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.5):
        super(BBM2_DeepNN, self).__init__()
        self.name = "BBM2-deep"

        self.input_size = input_size
        self.output_size = output_size

        # Layers
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 125)
        self.fc4 = nn.Linear(125, 9)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, full_tensor):
        # Process the input
        x = F.relu(self.fc1(full_tensor))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        output = self.fc4(x)
        return output
