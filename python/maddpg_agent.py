import numpy as np

from utils.memory_replay import ReplayBuffer
from utils.priority_replay import PriorityBuffer
from utils import model_enums
from ddpg_agent import Agent as ddpgAgent
from d4pg_agent import Agent as d4pgAgent

import torch

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = config.seed
        self.batch_size = config.BATCH_SIZE
        self.buffer_size = config.EXP_REPLAY_SIZE
        self.device = config.device
        self.update_every = config.UPDATE_FREQ
        self.LR_rate_decay = config.LR_rate_decay
        self.noise_decay = config.NOISE_DECAY
        self.learns_per_update = config.LEARNS_PER_STEP
        self.priority_replay = config.USE_PRIORITY_REPLAY
        self.priority_eps = config.PRIORITY_EPS
        self.model = config.model
           
        # Instantiate Multiple  Agent
        if config.model == model_enums.model_types.maddpg:
            self.agents = [ ddpgAgent(state_size, action_size, config, num_agents) 
                            for i in range(num_agents) ]
        else:
            self.agents = [ d4pgAgent(state_size, action_size, config, num_agents) 
                            for i in range(num_agents) ]
        
        # Instantiate Memory replay Buffer (shared between agents)
        if self.priority_replay:
            self.memory = PriorityBuffer(self.buffer_size, self.batch_size, self.device, self.seed, eps= 0.01, alpha=0.6, beta=0.4)
        else:
            self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device, self.seed)
        self.t_step = 0
        
    # resets all agents              
    def reset(self):
        for agent in self.agents:
            agent.sigma *= self.noise_decay
            agent.reset()

    def update_learning_rate(self):
        for agent in self.agents:
            agent.lr_actor *= self.LR_rate_decay
            agent.lr_critic *= self.LR_rate_decay
            agent.set_learning_rate(agent.lr_actor, agent.lr_critic)

    def act(self, states, addNoise):
        actions = []
        for agent_state, agent, agent_noise in zip(states, self.agents, addNoise):
            action = agent.act(agent_state, agent_noise)
            action = np.reshape(action, newshape=(-1))
            actions.append(action)
        actions = np.stack(actions)
        return actions
                
    # store states, actions, etc into ReplayBuffer and trigger training regularly
    # state and new_state : state of all agents [agent_no, state of an agent]
    # action: action of all agents [agent_no, action of an agent]
    # reward: reward of all agents [agent_no]
    # dones: dones of all agents [agent_no]
    def step(self, state, action, reward, next_state, done):  
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and self.memory.is_ready():
            for _ in range(self.learns_per_update):
                for agent in self.agents:
                    experiences = self.memory.sample(self.batch_size)
                    self.learn(experiences, agent)
    
    # execute learning on an agent    
    def learn(self, experiences, agent):
        # batch dataset for training
        states, actions, rewards, next_states, dones, ids, weights = experiences
        
        states = states.view(-1, self.num_agents, self.state_size)
        actions = actions.view(-1, self.num_agents, self.action_size)
        next_states = next_states.view(-1, self.num_agents, self.state_size)
        rewards = rewards.view(-1, self.num_agents)
        dones = dones.view(-1, self.num_agents)
   
        # forward each agent's actor target with their respectinve state
        actor_target_actions = torch.zeros(actions.shape, dtype=torch.float, device=self.device)
        for agent_idx, agent_i in enumerate(self.agents):
            if agent == agent_i:
                agent_id = agent_idx
            agent_i_current_state = states[:,agent_idx]
            actor_target_actions[:,agent_idx,:] = agent_i.actor_target.forward(agent_i_current_state)
        actor_target_actions = actor_target_actions.view(self.batch_size, -1)
     
        # replace action of the specific agent with actor_local output (NOISE removal)
        state = states[:,agent_id,:]
        actor_local_actions = actions.clone()
        actor_local_actions[:, agent_id, :] = agent.actor_local.forward(state)
        actor_local_actions = actor_local_actions.view(self.batch_size, -1)
        
        # get the agent's reward/done
        rewards = rewards[:,agent_id].view(-1,1)
        dones = dones[:,agent_id].view(-1,1)

        # flatten actions
        actions = actions.view(self.batch_size, -1)
        states =states.view(self.batch_size, -1)
        next_states =next_states.view(self.batch_size, -1)

        experiences = (states, actions, actor_local_actions, actor_target_actions, next_states, rewards, dones, weights)
        td_error = agent.learn(experiences)  

        if self.priority_replay:
            self.memory.updatePriorities(ids, abs(td_error))   
                        
    def saveCheckpoints(self, filepath):
        """Save checkpoints for all Agents"""
        for i in range(len(self.agents)):  
            agent = self.agents[i]        
            torch.save(agent.actor_local.state_dict(), filepath + str(i) + '_checkpoint_actor_local.pth') 
            torch.save(agent.critic_local.state_dict(), filepath + str(i) + '_checkpoint_critic_local.pth')             
            torch.save(agent.actor_target.state_dict(), filepath + str(i) +'_checkpoint_actor_target.pth') 
            torch.save(agent.critic_target.state_dict(), filepath + str(i) + '_checkpoint_critic_target.pth')

    def loadCheckpoints(self):
        if self.model == model_enums.model_types.mad4pg:
            for i in range(self.num_agents):  
                agent = self.agents[i]   
                agent.actor_local.load_state_dict(torch.load("../checkpoints/mad4pg/mad4pg_per" +str(i) + "_checkpoint_actor_local.pth"))
                agent.critic_local.load_state_dict(torch.load("../checkpoints/mad4pg/mad4pg_per" +str(i) + "_checkpoint_critic_local.pth"))     
                
        elif self.model == model_enums.model_types.maddpg:
            if self.priority_replay == True:
                for i in range(self.num_agents):  
                    agent = self.agents[i]   
                    agent.actor_local.load_state_dict(torch.load("../checkpoints/maddpg_per/maddpg_per" +str(i) + "_checkpoint_actor_local.pth"))
                    agent.critic_local.load_state_dict(torch.load("../checkpoints/maddpg_per/maddpg_per" +str(i) + "_checkpoint_critic_local.pth"))  
            else:
                for i in range(self.num_agents):  
                    agent = self.agents[i]   
                    agent.actor_local.load_state_dict(torch.load("../checkpoints/maddpg/maddpg" +str(i) + "_checkpoint_actor_local.pth"))
                    agent.critic_local.load_state_dict(torch.load("../checkpoints/maddpg/maddpg" +str(i) + "_checkpoint_critic_local.pth"))  
                    agent.actor_target.load_state_dict(torch.load("../checkpoints/maddpg/maddpg" +str(i) + "_checkpoint_actor_target.pth"))
                    agent.critic_target.load_state_dict(torch.load("../checkpoints/maddpg/maddpg" +str(i) + "_checkpoint_critic_target.pth"))  

            
        