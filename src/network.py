from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):

    # Initialize the multilayer perceptron
    def __init__(self, inp_size, out_size):
        super(MLP, self).__init__()

        self.input_layer = nn.Linear(inp_size, 64)
        self.hidden_layer = nn.linear(64, 64)
        self.output_layer = nn.Linear(64, out_size)

    # Completes one forward pass through the network 
    def forward(self, inp):
        
        activated_1 = F.relu(self.input_layer(inp))
        activated_2 = F.relu(self.hidden_layer(activated_1))

        return self.output_layer(activated_2)
