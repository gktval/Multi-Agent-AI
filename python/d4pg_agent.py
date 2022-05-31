import numpy as np

import model
from utils import noise,l2_projection,memory_replay,priority_replay
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config, num_agents):
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
        self.priority_eps = config.PRIORITY_EPS


        self.num_atoms = 51 # number of atoms in output layer of distributed critic
        self.v_max = 1 # upper bound of critic value output distribution
        self.v_min = -1 # lower bound of critic value output distribution
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.bin_centers = np.array([self.v_min+i*self.delta_z for i in range(self.num_atoms)]).reshape(-1,1)
           
        # Network
        self.actor_local = model.Actor(state_size, action_size).to(self.device)
        self.actor_target = model.Actor(state_size, action_size).to(self.device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(),lr=self.lr_actor)

        self.critic_local=model.CriticDist(state_size*num_agents,action_size*num_agents, self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.critic_target=model.CriticDist(state_size*num_agents,action_size*num_agents, self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.critic_optim=optim.Adam(self.critic_local.parameters(), lr=self.lr_critic) #, weight_decay=self.weight_decay )

        self.soft_update(self.critic_local, self.critic_target,1)
        self.soft_update(self.actor_local, self.actor_target,1)

        # Replay memory
        if num_agents == 1:
            if self.priority_replay:
                self.memory = priority_replay.PriorityBuffer(self.buffer_size, self.batch_size, self.device, self.seed, eps= 0.01, alpha=0.6, beta=0.4)
            else:
                self.memory = memory_replay.ReplayBuffer(self.buffer_size, self.batch_size, self.device)

        # Noise process
        if self.is_noisy:
            self.noise = noise.OUNoise()
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.step_count = 0

    def reset(self):
        if self.is_noisy:
            self.noise.reset(self.sigma)
            self.step_count += 1


    # Set learning rates
    def set_learning_rate(self, lr_actor, lr_critic):
        self.learning_rate_actor = lr_actor
        self.learning_rate_critic = lr_critic
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr_actor
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr_critic


    def step(self, states, actions, rewards, next_states, dones):
        # add experience to memory
        for state,action,reward,next_state,done in zip(states,actions,rewards,next_states,dones):
            self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and self.memory.is_ready():
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
    

    def act(self, states, addNoise):
        numpyState = torch.from_numpy(np.array(states))
        state = numpyState.float().to(self.device) 

        self.actor_local.eval()
        
        
        with torch.no_grad():
            action = self.actor_local(state) 
            
        action = action.cpu().detach().numpy()
        if addNoise == True:
             action += self.noise.sample()

        self.actor_local.train()

        return np.clip(action, -1, 1) # all actions between -1 and 1

        

    def learn(self, experiences):
        states, actions, actor_local_actions, actor_target_actions, next_states, rewards, dones, weights = experiences

         # ------------------- update critic ------------------- #       
        #next_actions = self.actor_target(next_states)
        target_pred = self.critic_target.forward(torch.cat((next_states, actor_target_actions), dim=1)).detach()

        # Get projected distribution
        proj_distr = l2_projection.distribution(next_distr_v=target_pred,
                                         rewards_v=rewards,
                                         dones_mask_t=dones,
                                         gamma=self.gamma ** self.n_step,
                                         n_atoms=self.num_atoms,
                                         v_min=self.v_min,
                                         v_max=self.v_max,
                                         delta_z=self.delta_z)
        proj_distr = torch.from_numpy(proj_distr).float().to(self.device)

        Q_expected = self.critic_local.forward(torch.cat((states, actions), dim=1))
        
        if self.priority_replay:
            prob_dist = F.mse_loss(Q_expected, proj_distr, reduction='none')
            critic_loss = prob_dist.mean(axis=1)
            td_error = critic_loss.cpu().detach().numpy().flatten()
            critic_loss = critic_loss * torch.tensor(weights).float().to(self.device)
            critic_loss = critic_loss.mean()
        else:
            critic_loss = F.mse_loss(Q_expected, proj_distr)
            td_error = None

        # Update step
        self.critic_optim.zero_grad()
        critic_loss.backward()
        #clip_grad_norm_(self.critic_local.parameters(), 10)
        self.critic_optim.step()

         # ------------------- update actor ------------------- #               
        #actions_actor = self.actor_local(states)
        actor_loss = self.critic_local.forward(torch.cat((states, actor_local_actions), dim=1))  
        actor_loss = actor_loss * torch.from_numpy(self.critic_local.z_atoms).float().to(self.device)
        # actor_loss = torch.sum(actor_loss, dim=1)
        actor_loss = actor_loss.mean(axis=1)
        actor_loss = -actor_loss.mean()
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

