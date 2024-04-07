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
        self.fc4 = nn.Linear(125, output_size)

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

class BBM3(nn.Module):
    def __init__(self, dropout=0.5):
        super(BBM3, self).__init__()
        self.name = "BBM3"

        self.input_size = 52
        self.output_size = 10

        # Layers
        self.fc1 = nn.Linear(52, 40)
        self.fc2 = nn.Linear(40, 30)
        self.fc3 = nn.Linear(30, 20)
        self.fc4 = nn.Linear(20, 10)

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

class BBM4(nn.Module):
    def __init__(self, dropout=0.5):
        super(BBM4, self).__init__()
        self.name = "BBM4"

        self.input_size = 52
        self.output_size = 10

        # Layers
        self.fc1 = nn.Linear(52, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 10)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, full_tensor):
        # Process the input
        x = F.relu(self.fc1(full_tensor))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output
