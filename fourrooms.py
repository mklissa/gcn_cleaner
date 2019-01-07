import numpy as np
from gym import core, spaces
from gym.envs.registration import register
import pdb

class Fourrooms:
    def __init__(self):
        layout = """\
wwwwwwwwwwwww
wsssssw     w
wsssssw     w
wsssss      w
wsssssw     w
wsssssw     w
ww wwww     w
w     www www
w    rw     w
w     w     w
w        g  w
w     w     w
wwwwwwwwwwwww
"""


#         layout = """\
# wwwwwwwwwwwwwwwwwwwwww
# wsssssssssw          w
# wsssssssssw          w
# wsssssssssw          w
# wsssssssssw   n      w
# wsssssssss           w
# wsssssssss           w
# wsssssssssw          w
# wsssssssssw          w
# wsssssssssw          w
# wwwwnnwwwwwwwww  wwwww
# w         w          w
# w         w          w
# w         w          w
# w         w          w
# w         w          w
# w              g     w
# w                    w
# w         w          w
# w         w          w
# w         w          w
# wwwwwwwwwwwwwwwwwwwwww
# """


#         layout = """\
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssss                      w
# wssssssssssssssssssss                      w
# wssssssssssssssssssss                      w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wwwwwwww     wwwwwwwwwwwwwwwww     wwwwwwwww
# w                    w                     w
# w                    w                     w
# w                    w                     w
# w                    w                     w
# w                                          w
# w                              g           w
# w                                          w
# w                                          w
# w                    w                     w
# w         r          w                     w
# w                    w                     w
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# """



#         layout = """\
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssss                      w
# wssssssssssssssssssss                      w
# wssssssssssssssssssss                      w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wsssssssssssssssssssswwwwwww   wwwwww. wwwww
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wssssssssssssssssssssw                     w
# wwwwwwww     wwwwwwwwwwwwwwwww     wwwwwwwww
# w                    w                     w
# w                    w                     w
# w                    w                     w
# wwww     wwwwwwwwwwwww                     w
# w                                          w
# w                              g           w
# w                                          w
# w                                          w
# w                    w                     w
# w                    w                     w
# w                    w                     w
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# """



#         layout = """\
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# w        ssssssssssssw                     w
# w        ssssssssssssw   wwwwwwwwwwwwww    w
# w    wwwwwwwwwwwwwwwww   w                 w
# w    w               w   w  wwwwwwwwww     w
# w    w    wwwwwwwwwwww   w           w     w
# w    w       w           wwwwwwwwwww w     w
# w    w  w    w   wwwww   w           w     w
# w    w  w    w   w   w   wwwwwwwwwwwww     w
# w    w  w    w   w   w               w     w
# w    w  w    w   w   wwwwwww   wwwwwwwwwwwww
# w       w    w   w   w     w   w           w
# w       w    w   w   w  w  w   w  wwwwww   w
# w    w  w    w   w   w  w  w   w  w        w
# w    w  w    w   w   w  w      w  w wwww   w
# w    w  w            w  w  w      w    w   w
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww wwwww
# w                    w                     w
# wwwwwwwwwwwwwwwww    w                     w
# w                    w                     w
# wwww     wwwwwwwwwwwww                     w
# w        w           w                     w
# w  w     w  w  w  w  w         g           w
# w  w     w  wwwwwwwwww                     w
# w  w     w           w                     w
# w  w     wwwwwwwww   w                     w
# w  w     w           w                     w
# w  w                 w                     w
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
# """


        self.grid = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        self.possible_states = np.argwhere((1. - self.grid).flatten() == 1)

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.grid == 0))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)


        

        self.tostate = {}
        statenum = 0
        for i in range(layout.count('\n')):
            for j in range(len(layout.splitlines()[0])):
                if self.grid[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum +=  1
        self.tocell = {v:k for k,v in self.tostate.items()}



        goal =np.array([list(map(lambda c: 1 if c=='g' else 0, line)) for line in layout.splitlines()]).flatten()

        
        # self.reward_labels = reward_labels = {'r':0.5,'n':-1.,'k':0.1}
        self.reward_labels = reward_labels = {'r':0.5,'n':-1.}
        more_rewards_states=[]
        more_rewards_labels=[]
        for line in layout.splitlines():
            more_rewards_states += list(map(lambda c: 1 if c in reward_labels.keys() else 0, line))
            for c in line:
                if c in reward_labels.keys():
                    more_rewards_labels += c



        init_states = np.array([list(map(lambda c: 1 if c=='s' else 0, line)) for line in layout.splitlines()]).flatten()

        self.grid2obs = grid2obs = dict(zip(np.argwhere(self.grid.flatten()==0).squeeze(),
                                                range(self.observation_space.n) ))

        # pdb.set_trace()
        self.init_states = map(grid2obs.get,np.argwhere(init_states == 1).flatten())
        self.goal = map(grid2obs.get,np.argwhere(goal == 1).flatten())[0]


        self.more_rewards_states = map(grid2obs.get,np.argwhere(np.array(more_rewards_states) == 1).flatten())
        self.more_rewards = dict(zip(self.more_rewards_states,list(map(reward_labels.get,more_rewards_labels))))

        self.original_rewards = self.more_rewards.copy()

        self.obs2grid = {v:k for k,v in grid2obs.items()}
        


    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.grid[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):

        # reset the possession of the key
        if 'k' in self.reward_labels:
            self.key = 0 # Only set key to 0 if we are playing with a key
        else:
            self.key = 1

        # reset the rewards
        for key in self.more_rewards.keys():
            self.more_rewards[key] = self.original_rewards[key]

        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        state = self.obs2grid.get(state)
        return state

    def step(self, action):
 
        # pdb.set_trace()

        if self.rng.uniform() < 0.:
            empty_cells = self.empty_around(self.currentcell)
            nextcell = empty_cells[self.rng.randint(len(empty_cells))]
        else:
            nextcell = tuple(self.currentcell + self.directions[action])

        if not self.grid[nextcell]:
            self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = False

        reward=0.
        if state == self.goal and self.key:
            reward = 1.
            done = state == self.goal
        elif state in self.more_rewards_states:
            reward = self.more_rewards.get(state)
            if reward == 0.1:
                self.key=1
            self.more_rewards[state] = 0.  # You can only catch the reward once per episode



        state = self.obs2grid.get(state)
        return state, reward, done, None
