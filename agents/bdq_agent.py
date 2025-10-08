import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F
from collections import deque
import time


class BDQNetwork(nn.Module):
    def __init__(self,
                 state_size,
                 branching_nodes,
                 hidden_size=128,
                 layers=2,
                 device="cpu"):
        """
        branching_nodes: dict mapping node -> candidate_count
        """

        super(BDQNetwork, self).__init__()
        self.device = device

        # Build shared hidden layers.
        # First layer: input -> hidden_size.
        self.shared_layers = nn.ModuleList()
        self.shared_layers.append(
            nn.Linear(state_size, hidden_size).to(device))
        # Additional layers: hidden_size -> hidden_size.
        for _ in range(layers - 1):
            self.shared_layers.append(
                nn.Linear(hidden_size, hidden_size).to(device))

        # state value branch
        self.value_stream = nn.Linear(hidden_size, 1).to(self.device)

        # Create one advantage stream per branching node with the correct number of outputs.
        self.branching_nodes = branching_nodes
        self.advantage_streams = nn.ModuleDict({
            str(node): nn.Linear(hidden_size, candidate_count).to(self.device)
            for node, candidate_count in branching_nodes.items()
        })

    def forward(self, state):
        x, V = self.get_value_stream(state)

        # advantages for branching nodes
        A_branches = {node: self.advantage_streams[str(node)](
            x) for node in self.branching_nodes}

        # normalised advantage - q-value per branch
        Q_branches = {node: V + (A - A.mean(dim=1, keepdim=True))
                      for node, A in A_branches.items()}

        return Q_branches

    def get_value_stream(self, state):
        # Pass state through all shared layers with ReLU activation.
        x = state
        for layer in self.shared_layers:
            x = F.relu(layer(x))

        V = self.value_stream(x)
        return x, V


class BDQAgent():
    def __init__(self,
                 state_size: int,
                 branching_nodes: dict,
                 candidate_splits: dict,
                 nnet_config: dict,
                 ):

        super().__init__()
        self.state_size = state_size
        self.branching_nodes = branching_nodes
        self.candidate_splits = candidate_splits

        # Set up device (GPU if available, else CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n\t--> Running model in {self.device}\n")

        # Build networks
        self.hidden_size = nnet_config["hidden_size"]
        self.layers = nnet_config["layers"]

        self.model = BDQNetwork(state_size, branching_nodes,
                                hidden_size=self.hidden_size,
                                layers=self.layers,
                                device=self.device)

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_branches = self.model(state_tensor)

        action_dict = {node: torch.argmax(
            q_branches[node]).item() for node in self.branching_nodes}

        return action_dict

    def get_action_split(self, action_dict):
        action_split = {}

        for branching_node in action_dict:
            split_idx = action_dict[branching_node]
            action_split[branching_node] = self.candidate_splits[branching_node][split_idx]

        return action_split
