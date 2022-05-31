import numpy as np

from utils.memory_replay import ReplayBuffer
from utils.priority_replay import PriorityBuffer
import model
from utils.noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config, num_agents = 1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = config.seed
        self.n_step = config.N_STEPS
        self.buffer_size = config.EXP_REPLAY_SIZE
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.batch_size = config.BATCH_SIZE
        self.update_every = config.UPDATE_FREQ
        self.lr_actor = config.LR_Actor
        self.lr_critic = config.LR_Critic
        self.priority_replay = config.USE_PRIORITY_REPLAY
        self.is_noisy = config.USE_NOISY_NETS
        self.device = config.device
        self.weight_decay = config.WEIGHT_DECAY
        self.sigma = config.SIGMA_INIT

           
        # Network
        self.actor_local = model.Actor(state_size, action_size).to(self.device)
        self.actor_target = model.Actor(state_size, action_size).to(self.device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(),lr=self.lr_actor)

        self.critic_local=model.Critic(state_size*num_agents,action_size*num_agents).to(self.device)
        self.critic_target=model.Critic(state_size*num_agents,action_size*num_agents).to(self.device)
        self.critic_optim=optim.Adam(self.critic_local.parameters(), lr=self.lr_critic) #, weight_decay=self.weight_decay )

        self.soft_update(self.critic_local, self.critic_target,1)
        self.soft_update(self.actor_local, self.actor_target,1)

        # Replay memory : in MADDPG, the ReplayBuffer is common to all agents
        if num_agents == 1:
            if self.priority_replay:
                self.memory = PriorityBuffer(self.buffer_size, self.batch_size, self.device, self.seed, eps= 0.01, alpha=0.6, beta=0.4)
            else:
                self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device, self.seed)

        # Noise process
        if self.is_noisy:
            self.noise = OUNoise()
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def reset(self):
        if self.is_noisy:
            self.noise.reset(self.sigma)

    # Set methods
    def set_learning_rate(self, lr_actor, lr_critic):
        self.learning_rate_actor = lr_actor
        self.learning_rate_critic = lr_critic
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr_actor
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr_critic   

    def act(self, states, addNoise):
        numpyState = torch.from_numpy(np.array(states))
        state = numpyState.float().to(self.device) 

        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state)
            
        if addNoise == True:
             action += self.noise.sample()

        self.actor_local.train()
        return np.clip(action.cpu().numpy().squeeze(), -1, 1) # all actions between -1 and 1
    

    def step(self, states, actions, rewards, next_states, dones):
        # add experience to memory
        for state,action,reward,next_state,done in zip(states,actions,rewards,next_states,dones):
            self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and self.memory.is_ready():
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)


    def learn(self, experiences):
        states, actions, actor_local_actions, actor_target_actions, next_states, rewards, dones, weights = experiences

         # ------------------- update critic ------------------- #       
        #next_actions = self.actor_target(next_states)
        pred = self.critic_target.forward(torch.cat((next_states, actor_target_actions), dim=1)).detach()
        Q_targets = rewards + (self.gamma**self.n_step) * pred * (1 - dones)
        Q_expected = self.critic_local.forward(torch.cat((states, actions), dim=1))
        
        # Compute loss
        if self.priority_replay:
            critic_loss = F.mse_loss(Q_expected, Q_targets,reduction='none')  
            td_error = critic_loss.cpu().detach().numpy().flatten()
            critic_loss= critic_loss.mean()
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            td_error = None

        self.critic_optim.zero_grad()
        critic_loss.backward()
        #clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()

         # ------------------- update actor ------------------- #               
        #actions_actor = self.actor_local(states)
        actor_loss = -self.critic_local.forward(torch.cat((states, actor_local_actions), dim=1)).mean()  
        self.actor_optim.zero_grad()   
        actor_loss.backward()
        #clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optim.step()

        # ------------------- update target networks ------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)  
        self.soft_update(self.actor_local, self.actor_target, self.tau)     

        return td_error
     
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)