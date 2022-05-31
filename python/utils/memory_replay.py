import random
import numpy as np
import torch

from collections import namedtuple
from collections import deque

from utils import buffer

from IPython.core.debugger import set_trace

class ReplayBuffer( buffer.IBuffer ) :

    def __init__( self, bufferSize, batchSize, device, randomSeed ) :
        super( ReplayBuffer, self ).__init__( bufferSize, batchSize, device, randomSeed )

        self._experience = namedtuple( 'Step', 
                                       field_names = [ 'state', 
                                                       'action',
                                                       'reward',
                                                       'nextState',
                                                       'endFlag' ] )

        self._memory = deque( maxlen = bufferSize )

        # seed random generator (@TODO: What is the behav. with multi-agents?)
        random.seed( randomSeed )

    def add( self, state, action, reward, nextState, endFlag ) :
        # create a experience object from the arguments
        _expObj = self._experience( state, action, reward, nextState, endFlag )
        # and add it to the deque memory
        self._memory.append( _expObj )

    def sample( self, batchSize ) :
        # grab a batch from the deque memory
        _expBatch = random.sample( self._memory, batchSize )

        _states = torch.from_numpy(np.vstack([e.state for e in _expBatch if e is not None])).float().to(self._device)
        _actions = torch.from_numpy(np.vstack([e.action for e in _expBatch if e is not None])).float().to(self._device)
        _rewards = torch.from_numpy(np.vstack([e.reward for e in _expBatch if e is not None])).float().to(self._device)
        _nextStates = torch.from_numpy(np.vstack([e.nextState for e in _expBatch if e is not None])).float().to(self._device)
        _endFlags = torch.from_numpy(np.vstack([e.endFlag for e in _expBatch if e is not None]).astype(np.uint8)).float().to(self._device)
        _indices = None
        _weights = None

        _endFlags_flatten = _endFlags.view(_endFlags.numel(), -1)
        _rewards_flatten = _rewards.view(_rewards.numel(), -1)

        return _states, _actions, _rewards_flatten, _nextStates, _endFlags_flatten, _indices, _weights

    def reinforce(self, count, reward):
        pres_mem = []
        for _ in range(count):   
            pres_mem.append(self._memory.pop())

        for mem in pres_mem:
            self.add(mem.state, mem.action, reward, mem.nextState, mem.endFlag ) 

    def is_ready(self):
        return len(self) >= self._batchSize

    def __len__( self ) :
        return len( self._memory )