from math import fabs
import torch
from maddpg_agent import Agent
from collections import deque
import numpy as np
import os
import time

def Run(env, config, n_episodes=2000):   

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # number of actions
    action_size = brain.vector_action_space_size

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.GAMMA = 0.99
    config.LR_Actor = 1e-3
    config.LR_Critic = 2e-3
    config.TAU = 1e-3 
    config.seed = 0
    config.USE_NOISY_NETS = True
    config.UPDATE_FREQ = 1
    config.BATCH_SIZE = 256
    config.LEARNS_PER_STEP = 3
    config.NOISE_DECAY = 0.985
    config.USE_PRIORITY_REPLAY = True
    if config.USE_PRIORITY_REPLAY:
        config.EXP_REPLAY_SIZE = 65536
    else:
        config.EXP_REPLAY_SIZE = 100000
    
    agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, config=config)
    scores_window = deque(maxlen=100)  # last 100 scores 
    total_scores = []
    max_score = 0
    add_noise = [True, True]
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations            # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        agent.reset()
        start = time.time()
        frames = 0

        while True:
            actions = agent.act(states, add_noise)              

            env_info = env.step(actions)[brain_name]        # send the action to the environment
            next_states = env_info.vector_observations   # get the next state
            rewards = env_info.rewards                   # get the reward
            dones = env_info.local_done                 # see if episode has finished
            scores += rewards                       # update the score (for each agent)  

            agent.step(states, actions, rewards, next_states, dones)       # step through agent learning  
            
            states = next_states                      # update the next states
            
            frames += 1
            if np.any(dones):                                  # exit loop if episode finished
                break

        elapsed_time = time.time() - start     
             
        mean = np.mean(np.max(scores))
        scores_window.append(mean)       # save most recent score
        total_scores.append(mean)       # update the total scores
        avg_score = np.mean(scores_window)

        print('\rEpisode {}\tCurrent Score: {:.2f}\tAverage Score: {:.2f}\tMax Score: {:.2f}\tTime: {:.2f}\tFrames: {:.2f} '
              .format(i_episode, mean, avg_score, max_score, elapsed_time, frames), end="")

        agent.update_learning_rate()
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))

        if mean > max_score:    
            max_score = mean
            if avg_score > 0.5:
                per = ""
                if config.USE_PRIORITY_REPLAY == True:
                    per = "_per"
                saveCheckpoints(str(config.model.name) + per, agent, config)

    env.close()
    return total_scores, scores_window

def saveCheckpoints(filename, agent, config):
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    agent.saveCheckpoints(config.model_path + filename)    

def watchAgent(config, env):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    num_agents = len(env_info.agents)

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.GAMMA = 0.99
    config.LR_Actor = 1e-3
    config.LR_Critic = 2e-3
    config.TAU = 1e-3 
    config.seed = 0
    config.USE_NOISY_NETS = True
    config.UPDATE_FREQ = 1
    config.BATCH_SIZE = 256
    config.LEARNS_PER_STEP = 3
    config.NOISE_DECAY = 0.985
    config.USE_PRIORITY_REPLAY = False
    config.USE_PRIORITY_REPLAY = True
    if config.USE_PRIORITY_REPLAY:
        config.EXP_REPLAY_SIZE = 65536
    else:
        config.EXP_REPLAY_SIZE = 100000
    
    agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, config=config)
    agent.loadCheckpoints()

    scores = np.zeros(num_agents)

    while True:
        action = agent.act(states, [False, False])
                
        env_info = env.step(action)[brain_name]        # send the action to the environment
        scores = env_info.rewards
        done = env_info.local_done                  # see if episode has finished
        if np.any(done):
            if np.mean(scores)>2:
                break

            
    env.close()
