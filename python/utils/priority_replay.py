import random
import numpy as np
import torch

from collections import namedtuple
from utils import segmentree
from utils import buffer

class PriorityBuffer( buffer.IBuffer ) :

    def __init__( self, 
                  bufferSize, 
                  batchSize,
                  device,
                  randomSeed, 
                  eps = 0.01, 
                  alpha = 0.6, 
                  beta = 0.4,
                  dbeta = 0.00001 ) :

        super( PriorityBuffer, self ).__init__( bufferSize, batchSize, device, randomSeed )

        # hyperparameters of Prioritized experience replay
        self._eps   = eps    # extra ammount added to the abs(tderror) to avoid zero probs.
        self._alpha = alpha  # regulates how much the priority affects the probs of sampling
        self._beta  = beta   # regulates how much away from true importance sampling we go (up annealed to 1)
        self._dbeta = dbeta  # regulates how much we anneal up the previous regulator of importance sampling

        # a handy experience tuple constructor
        self._experience = namedtuple( 'Step', 
                                       field_names = [ 'state', 
                                                       'action',
                                                       'reward',
                                                       'nextState',
                                                       'endFlag' ] )
        # sumtree for taking the appropriate samples
        self._sumtree = segmentree.SumTree( self._bufferSize )
        # mintree for taking the actual min as we go
        self._mintree = segmentree.MinTree( self._bufferSize )

        # a variable to store the running max priority
        self._maxpriority = 1.
        # a variable to store the running min priority
        self._minpriority = eps

        # number of "actual" elements in the buffer
        self._count = 0

        ## # @TEST: Forcing to be equal to no priority
        ## self._alpha = 0.
        ## self._eps = 0.

        ## # seed random generator (@TODO: What is the behav. with multi-agents?)
        ## random.seed( randomSeed ) # no need to seed this generator. Using seeded np generator

    def add( self, state, action, reward, nextState, endFlag ) :
        """Adds an experience tuple to memory
        Args:
            state (np.ndarray)      : state of the environment at time (t)
            action (int)            : action taken at time (t)
            nextState (np.ndarray)  : state of the environment at time (t+1) after taking action
            reward (float)          : reward at time (t+1) for this transition
            endFlag (bool)          : flag that indicates if this state (t+1) is terminal
        """

        # create a experience object from the arguments
        _expObj = self._experience( state, action, reward, nextState, endFlag )

        # store the data into a node in the smtree, with nodevalue equal its priority
        # maxpriority is used here, to ensure these tuples can be sampled later
        self._sumtree.add( _expObj, self._maxpriority ** self._alpha )
        self._mintree.add( _expObj, self._maxpriority ** self._alpha )

        # update actual number of elements
        self._count = min( self._count + 1, self._bufferSize )

    def sample( self, batchSize ) :
        """Samples a batch of data using consecutive sampling intervals over the sumtree ranges
        Args:
            batchSize (int) : number of experience tuples to grab from memory
        Returns:
            (indices, experiences) : a tuple of indices (for later updates) and 
                                     experiences from memory
        Example:
                    29
                   /  \
                  /    \
                 13     16        
                |  |   |  |
               3  10  12  4       |---| |----------| |------------| |----|
                                    3        10            12          4
                                  ^______^______^_______^_______^_______^
                                        *      *      *   *    *     *
            5 samples using intervals, and got 10, 10, 12, 12, 4
        """
        # experiences sampled, indices and importance sampling weights
        _expBatch = []
        _indicesBatch = []
        _impSampWeightsBatch = []

        # compute intervals sizes for sampling
        _prioritySegSize = self._sumtree.sum() / batchSize

        # min node-value (priority) in sumtree
        _minPriority = self._mintree.min()
        # min probability that a node can have
        _minProb = _minPriority / self._sumtree.sum()

        # take sampls using segments over the total range of the sumtree
        for i in range( batchSize ) :
            # left and right ticks of the segments
            _a = _prioritySegSize * i
            _b = _prioritySegSize * ( i + 1 )
            ## _b = min( _prioritySegSize * ( i + 1 ), self._sumtree.sum() - 1e-5 )

            # throw the dice over this segment
            _v = np.random.uniform( _a, _b )
            _indx, _priority, _exp = self._sumtree.getNode( _v )

            # Recall how importance sampling weight is computed (from paper)
            #
            # E   { r } = E    { p * r  } = E     { w * r }  -> w : importance 
            #  r~p         r~p'  -           r~p'                   sampling
            #                    p'                                 weight
            #
            # in our case:
            #  p  -> uniform distribution
            #  p' -> distribution induced by the priorities
            #
            #      1 / N
            #  w = ____   
            #
            #       P(i) -> given by sumtree (priority / total)
            #
            # for stability, the authors scale the weight by the max-weight ...
            # possible, which is (because maximizing a fraction minimizes the ...
            # denominator if the numrerator is constant=1/N) the weight of the ...
            # node with minimum probability. After some operations :
            # 
            #                          b                     b
            # w / wmax = ((1/N) / P(i))   / ((1/N) / minP(j))   
            #                                          j
            #                               b                      -b
            # w / wmax = ( min P(j) / P(i) )   = ( P(i) / min P(j) )
            #               j                              j

            # compute importance sampling weights
            _prob = _priority / self._sumtree.sum()
            _impSampWeight = ( _prob / _minProb ) ** -self._beta

            # accumulate in batch
            _expBatch.append( _exp )
            _indicesBatch.append( _indx )
            _impSampWeightsBatch.append( _impSampWeight )

        # stack each experience component along batch axis
        _states = torch.from_numpy(np.vstack([e.state for e in _expBatch if e is not None])).float().to(self._device)
        _actions = torch.from_numpy(np.vstack([e.action for e in _expBatch if e is not None])).float().to(self._device)
        _rewards = torch.from_numpy(np.vstack([e.reward for e in _expBatch if e is not None])).float().to(self._device)
        _nextStates = torch.from_numpy(np.vstack([e.nextState for e in _expBatch if e is not None])).float().to(self._device)
        _endFlags = torch.from_numpy(np.vstack([e.endFlag for e in _expBatch if e is not None]).astype(np.uint8)).float().to(self._device)

        # convert indices and importance sampling weights to numpy-friendly data
        _indicesBatch = np.array( _indicesBatch ).astype( np.int64 )
        _impSampWeightsBatch = np.array( _impSampWeightsBatch ).astype( np.float32 )

        # anneal the beta parameter
        self._beta = min( 1., self._beta + self._dbeta )

        _endFlags_flatten = _endFlags.view(_endFlags.numel(), -1)
        _rewards_flatten = _rewards.view(_rewards.numel(), -1)

        return _states, _actions, _rewards_flatten, _nextStates, _endFlags_flatten, _indicesBatch, _impSampWeightsBatch

    def updatePriorities( self, indices, absBellmanErrors ) :
        """Updates the priorities (node-values) of the sumtree with new bellman-errors
        Args:
            indices (np.ndarray)        : indices in the sumtree that have to be updated
            bellmanErrors (np.ndarray)  : bellman errors to be used for new priorities
        """
        # sanity-check: indices bath and bellmanErrors batch should be same length
        assert ( len( indices ) == len( absBellmanErrors ) ), \
               'ERROR> indices and bellman-errors batch must have same size'

        # add the 'e' term to avoid 0s
        _priorities = np.power( absBellmanErrors + self._eps, self._alpha )

        for i in range( len( indices ) ) : 
            # update each node in the sumtree and mintree
            self._sumtree.update( indices[i], _priorities[i] )
            self._mintree.update( indices[i], _priorities[i] )
            # update the max priority
            self._maxpriority = max( _priorities[i], self._maxpriority )

    def is_ready(self):
        return len(self) >= self._batchSize

    def __len__( self ) :
        return self._count

    @property
    def alpha( self ) :
        return self._alpha

    @property
    def beta( self ) :
        return self._beta    

    @property
    def eps( self ) :
        return self._eps