import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torch.distributions import Normal
import numpy as np
from math import *

def weight_init(layers):
    for layer in layers:
        nn.init.xavier_uniform_(layer.weight)


class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        super(Actor, self).__init__()

        #self.seed = torch.manual_seed(seed) 
       
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        weight_init([self.fc1, self.fc2, self.fc3])
         
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(state))   
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x

    
class Critic(nn.Module):
    """Critic (Value) Model."""
    
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=256, fc3_units = 128, fc4_units = 32):
        super(Critic, self).__init__()
    
        #self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.Q = nn.Linear(fc4_units, 1)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        weight_init([self.fc1, self.fc2, self.fc3, self.fc4, self.Q])


    def forward(self,state):
        """Build an critic (value) network that maps (states,actions) pairs to Q-values."""
        
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.Q(x)
            
        return x  

class CriticDist(nn.Module):
    """Critic (Value) Model."""
    
    def __init__(self, state_size, action_size, v_min, v_max, num_atoms, 
                fc1_units=256, fc2_units=256, fc3_units = 128, fc4_units = 32):
        super(CriticDist, self).__init__()
    
        #self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.Q = nn.Linear(fc4_units, num_atoms)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        weight_init([self.fc1, self.fc2, self.fc3, self.fc4, self.Q])

        self.z_atoms = np.linspace(v_min, v_max, num_atoms)

    def forward(self,state):
        """Build an critic (value) network that maps (states,actions) pairs to Q-values."""
        
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.Q(x)
            
        return x  

    # def forward(self, state, action):
    #     """Build an critic (value) network that maps (states,actions) pairs to Q-values."""
        
    #     x = torch.cat([state, action], 1)
    #     x = F.relu(self.fc1(x))
    #     x = self.bn1(x)
    #     x = F.relu(self.fc2(x))
    #     x = F.relu(self.fc3(x))
    #     x = F.relu(self.fc4(x))
    #     x = self.Q(x)

    #     return x

    # def get_probs(self, state, action):
    #     return torch.softmax(self.forward(state, action), dim=1)
   

class GaussianActorCriticNet(nn.Module):

    def __init__(self, state_size, action_size, shared_layers, actor_layers, critic_layers, std_init=0):
        super(GaussianActorCriticNet, self).__init__()

        self.shared_network = GaussianActorCriticNet._create_shared_network(state_size, shared_layers)
        shared_output_size = state_size if len(shared_layers) == 0 else shared_layers[-1]

        self.actor_network = GaussianActorCriticNet._create_actor_network(shared_output_size, actor_layers, action_size)
        self.critic_network = GaussianActorCriticNet._create_critic_network(shared_output_size, critic_layers)

        self.std = nn.Parameter(torch.ones(action_size) * std_init)


    @staticmethod
    def _create_shared_network(state_size, shared_layers):
        iterator = chain([state_size], shared_layers)

        last_size = None
        args = []
        for layer_size in iterator:
            if last_size is not None:
                args.append(nn.Linear(last_size, layer_size))
                args.append(nn.ReLU())
            last_size = layer_size

        return nn.Sequential(*args)

    @staticmethod
    def _create_actor_network(input_size, actor_layers, action_size):
        iterator = chain([input_size], actor_layers, [action_size])

        last_size = None
        args = []
        for layer_size in iterator:
            if last_size is not None:
                args.append(nn.Linear(last_size, layer_size))
                args.append(nn.ReLU())
            last_size = layer_size

        # Replace last ReLU layer with tanh
        del args[-1]
        args.append(nn.Tanh())

        return nn.Sequential(*args)

    @staticmethod
    def _create_critic_network(input_size, critic_layers):
        iterator = chain([input_size], critic_layers, [1])

        last_size = None
        args = []
        for layer_size in iterator:
            if last_size is not None:
                args.append(nn.Linear(last_size, layer_size))
                args.append(nn.ReLU())
            last_size = layer_size

        # Remove last ReLU layer
        del args[-1]

        return nn.Sequential(*args)

    def forward(self, states, action=None, std_scale=1.0):
        shared_output = self.shared_network(states)
        mu = self.actor_network(shared_output)
        value = self.critic_network(shared_output)

        distribution = Normal(mu, std_scale * F.softplus(self.std))
        if action is None:
            action = distribution.sample() if std_scale > 0 else mu

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy().sum(-1).unsqueeze(-1)

        value = value.squeeze(1)

        return value, action, log_prob, entropy