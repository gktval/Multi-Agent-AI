## REPORT

### Introduction
This project implements and compares three networks for training agents in Unity's Tennis environment. Two networks were tested MADDMP and MA4DPG.

### 1. MADDPG Algorithm
The environments was solved initially with a Multi Agent Deep Deterministic Policy Gradient (MADDPG) algorithm as discussed in this paper https://arxiv.org/pdf/1509.02971.pdf. This base of this model came from project #2 in Udacity's learning program: https://github.com/gktval/Continuous-Control-AI. 

There are 2 agents, each with their own model, states and actions. During each iteration, the states are collected from the environment and passed two each agent. Each state consists of 24 state spaces. Of those 24 state spaces, 8 of them are from the current game frame, and 8 each from the two previous frames.

The two agents share a replay buffer. During the learning process, actor local actions and the actor target actions are combined for both agents and passed into each of the agents' models. Each agent's critical model will use the combined actor target actions to compute the gradient loss. Likewise, the agent's actor model will compute the gradient loss based on both agent's local actor actions.

The model for the MADDPG is discussed in this paper: https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf

![Pong](multi-agent-actor-critic.png)"Figure 1: Multi-agent decentralized actor with centralized critic (Lowe and Wu et al)."


For each step, each agent performs 3 updates. This greatly reduced the number of iterations needed for the agents to learn, but also increased the learning time threefold. 

Gradient clipping was tested on the model, but the results (not shown) were not as desirable as without. Other tests included using different learning rates, how often the models were updated, and the noise sigma. I also tested a variable learning update method based on whether or not the iteration was successful or not. In the end, I found that these parameters were optimal for my agent's learning:

	Gamma: 0.99
	Actor Learning Rate: 1e-3
	Critic Learning Rate: 2e-3
	Tau: 1e-3 
	Update Frequency: 1
	Batch Size: 256
	Learns per step:  3
	Noisy Net: Ornstein-Uhlenbeck process
	Noise Decay: 0.985
	Buffer Size: 100000

The MADDPG model was also tested with a priority replay buffer. In the report graph showing the results, there are two lines displaying the MADDPG with priority replay. The first time the model was ran, the learning process was prematurely stopped at 1500 episodes. The second time through, it ran to the full 2000 episodes. Interestingly though, the results varied in the number of episodes it took to complete the challenge (a 150 episode difference).


### 2. MAD4PG Algorithm
The Multi Agent Distributed Distributional Deterministic Policy Gradient (MAD4PG) algorithm is very similar to the MADDPG algorithm with a few exceptions: it uses a prioritized replay buffer, N-step returns, and a distributional critical update. The paper detailing D4PG can be found here: https://openreview.net/forum?id=SyZipzbCb. The paper discusses combining categorical and gaussian operations to predict the critic. In my code, only a categorical operation is achieved on the critic. The atom size was 51, which is used to compute the distributions of the final output from the critic network.

Below are the hyperparameters for training the MAD4PG network:

	Gamma: 0.99
	Actor Learning Rate: 1e-3
	Critic Learning Rate: 2e-3
	Tau: 1e-3 
	Update Frequency: 1
	Batch Size: 256
	Learns per step:  3
	Noisy Net: Ornstein-Uhlenbeck process
	Noise Decay: 0.985
	Buffer Size: 100000
	Tau = 0.001
	N-steps = 5
	Priority Epsilon = .0001
	N_Atoms = 51
	Vmin = -1
	Vmax = 1



### 3. Models for MADDPG and MAD4PG
I wanted the comparison between DDPG and D4PG as close as possible for comparisons. Thus, the models were nearly identical between the two. 

	Actor 
		Hidden 1: (input, 128) - ReLU
		Hidden 2: (128, 128) - ReLU
		Output: (128, 4) - TanH

	Critic
		Hidden 1: (input, 128) - Linear
		Hidden 2: (128, 256) - Linear
		Hidden 3: (256, 128) - Linear
		Hidden 4: (128, 32) - Linear
		Output: (32, 1 [51]) - Linear

In the MAD4PG model, the output was 51, which is equal to the number of atoms for the distribution update.


### Results
The results of the training will be stored in a folder called `scores` location in the `python` folder. After running several of the deep neural networks, you can replay the trained agent by changing the `isTest` variable passed into the `run()` method. 

Each of the algorithms described above achieved an average score of +0.5 over 100 episodes as listed below:

	MADDPG - 845 episodes
	MADDPG with PER (1) - 792
	MADDPG with PER (2) - 691
	MAD4PG - 1061 episodes

Both MADDPG models using priority replay performed better than MADDPG without the priority replay buffer. A lot variables were testing to make the MA4DPG algorithm perform well, but ultimately, the best I could accomplish was completing the environment in 1061 episodes. This model also had the lowest average scores and seemed to progressively do worse as more episodes were completed.


The plot of rewards can be found here:
[https://github.com/gktval/Multi-Agent_AI/blob/main/results.png](https://github.com/gktval/Multi-Agent-AI/blob/main/results.png)

The scores from each agent can be found here:
[https://github.com/gktval/Multi-Agent_AI/blob/main/python/scores/](https://github.com/gktval/Multi-Agent-AI/tree/main/scores)

The replay from each agent can be found here:
[https://github.com/gktval/Multi-Agent_AI/blob/main/python/checkpoints/](https://github.com/gktval/Multi-Agent-AI/tree/main/checkpoints)

### Future work
Future work could add other multi-agent algorithms, such as MAPPO. Notably, there were several episodes where the agents would perform a top score of 2.7, but then the next episode they would have a combined average score of 0. As a result, future work should be directed to determine why the agent's scores varied considerably from one episode to the next and how to reduce that. 

Furthermore, as presented in the D4PG research article, a gaussian distribution could be used to improve the loss. A lot of work is needed to improve the fine tuning of the model and the efficiency of the D4PG model.
