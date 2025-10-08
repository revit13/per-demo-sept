from config import ENVIRONMENT, NNETWORKS
from agents.bdq_agent import BDQAgent

import torch
import numpy as np
import random
import os
import time
import json

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)                # CPU
torch.cuda.manual_seed(seed)           # Current GPU
torch.cuda.manual_seed_all(seed)

initial_state = np.load("data/initial_state.npy")

# Load model and agent
with open("data/branching_nodes_candidate_splits.json", "r") as f:
    data = json.load(f)

branching_nodes = {int(k): v for k, v in data["branching_nodes"].items()}
candidate_splits = {int(k): v for k, v in data["candidate_splits"].items()}

agent = BDQAgent(state_size=initial_state.flatten().size,
                 branching_nodes=branching_nodes,
                 candidate_splits=candidate_splits,
                 nnet_config=NNETWORKS)

######################
saved_model_path = './model/' + ENVIRONMENT['map_name'] + '_model.pt'
checkpoint = torch.load(saved_model_path, map_location="cpu")

agent.model.load_state_dict(checkpoint["model_state_dict"])
agent.model.to(agent.device)
######################

# Run inference
initial_state_flatten = initial_state.flatten()
action = agent.act(initial_state_flatten)

print(f'Action: {action}')
