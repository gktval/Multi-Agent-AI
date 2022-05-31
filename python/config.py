import torch
import math
from utils.model_enums import model_types

class Config(object):
    def __init__(self):
        #Device type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model
        self.model = model_types.maddpg
        self.model_path = "../checkpoints/"

        #Random seed
        self.seed = 0

        #algorithm control
        self.USE_NOISY_NETS=False
        self.USE_PRIORITY_REPLAY=False
        
        #Multi-step returns
        self.N_STEPS = 1
        self.LEARNS_PER_UPDATE = 1

        #misc agent variables
        self.GAMMA=0.99 # discount factor
        self.LR_Actor=1e-3 # learning rate 
        self.LR_Critic=1e-3 # critic learning rate 
        self.LR_rate_decay = .999
        self.TAU = 1e-3 # for soft update of target parameters
        self.PRIORITY_EPS = 1e-4
        self.WEIGHT_DECAY=0 # decay the weight
        self.EPS_CLIP = .2
        self.OPT_STEPS = 16
        self.OPT_EPS = 5e-4
        self.GAE_LAMBDA=0.9
        self.GRAD_CLIP = 0.25
        self.ENT_PENALTY = 0.01
        self.LOSS_WEIGHT = 1

        #memory
        self.EXP_REPLAY_SIZE = 1024
        self.BATCH_SIZE = 128
        self.PRIORITY_ALPHA=0.6
        self.PRIORITY_BETA_START=0.4

        #Noisy Nets
        self.NOISE_DECAY = 0.995
        self.SIGMA_INIT = 0.5

        #Learning control variables
        self.LEARN_START = 10000
        self.MAX_FRAMES=100000
        self.UPDATE_FREQ = 1 # how often to update the network
        self.LEARNS_PER_STEP = 1 # how many times the network is updated in a single step
