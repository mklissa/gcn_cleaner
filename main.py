import gym
import argparse
import numpy as np
from fourrooms import Fourrooms
from utils import *
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
import os 
import networkx as nx
from graph import *

import pdb
from scipy.special import expit
from scipy.misc import logsumexp

colors = [(0,0,0)] +[(0.5,0.5,0.5)]+ [(cm.viridis(i)) for i in xrange(2,256)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


plotsteps=0



flags.DEFINE_integer('gen', 1, 'Do you want to generate a graph?')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 1e-2, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('nf', 1, 'Create features or not.')
flags.DEFINE_integer('f', 0, 'Create features or not.')
flags.DEFINE_string('fig', '', 'Figure identifier.')
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('app', '', 'For data file loading') 
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('eachepoch', 0, 'Plot difusion at each epoch or not.')
flags.DEFINE_integer('path', 0, 'Plot each epochs path.')




flags.DEFINE_integer('ngraph', 10, "Number of episodes before graph generation")
flags.DEFINE_integer('nepisodes', 500, "Number of episodes per run")
flags.DEFINE_integer('nruns',1, "Number of runs")
flags.DEFINE_integer('nsteps',1000, "Maximum number of steps per episode")
flags.DEFINE_integer('noptions',1, 'Number of options')
flags.DEFINE_integer('baseline',1, "Use the baseline for the intra-option gradient")
flags.DEFINE_integer('primitive',0, "Augment with primitive")



flags.DEFINE_float('temperature',1e-1, "Temperature parameter for softmax")
flags.DEFINE_float('discount',0.99, 'Discount factor')
flags.DEFINE_float('lr_intra',1e-1, "Intra-option gradient learning rate")
flags.DEFINE_float('lr_critic',1e-1, "Learning rate")




flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 46, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden5', 8, 'Number of units in hidden layer 1.')




want_graph= FLAGS.gen
seeds = [FLAGS.seed]
totalsteps = []

for seed in seeds:
    print('seed:',seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    rng = np.random.RandomState(seed)

    
    env = Fourrooms()
    walls = np.argwhere(env.grid.flatten()==1)
    row,col = env.grid.shape
    # pdb.set_trace()
    features = Tabular(env.observation_space.n)
    nfeatures, nactions = len(features), env.action_space.n


    observations = set()
    G = nx.Graph()
    G.add_nodes_from(range(len(env.grid.flatten())))


    options_grid2obs = [env.grid2obs, ]

    option_policies = [SoftmaxPolicy(rng, nfeatures, nactions, FLAGS.temperature,FLAGS.lr_intra), ]

    critics = [StateValue(FLAGS.discount, FLAGS.lr_critic, np.zeros((nfeatures)) ), ]
    action_critics = [ActionValue(FLAGS.discount, FLAGS.lr_critic, np.zeros((nfeatures,nactions)) ), ]

    done=False
    cumsteps = 0.
    optionsteps = 0.
    myrand = 1.
    myrandinit =1.
    sources=[]
    reward_states = set()

    init_set=[]
    goals = []
    allstates=[]
    for episode in range(FLAGS.nepisodes):
        epoch_states= set()
        rewards = []
        pos = env.grid.flatten().astype(float)
        pos[pos == 1] = -.5
        
        observation = env.reset()
        # start = env.obs2grid.get(observation)
        start=observation
        sources.append(start)
        observations.add(start)


        if init_set:
            option=1
        else:
            option=0

        newpos = options_grid2obs[option].get(observation)
        newobs = observation
        if newpos is None and newobs not in walls:
            allstates.append(newobs)
            critics[option].add(np.zeros((1)) )
            action_critics[option].add(np.zeros((1,4)) )
            option_policies[option].add(np.zeros((1,4)) )
            options_grid2obs[option][newobs] = len(critics[option].weights) - 1



        last_phi = phi = options_grid2obs[option].get(observation)

        action = option_policies[option].sample(phi,0)  if np.random.rand() > 0.1 else np.random.randint(4)
        critics[option].save(phi)
        action_critics[option].save(phi, action)
        option_policies[option].save(phi,action)

        
        cumreward = 0.
        for step in range(FLAGS.nsteps):



            acts = [-col,col,-1,1]
            newpos = options_grid2obs[option].get(observation+acts[action])
            newobs = observation+acts[action]
            if newpos is None and newobs not in walls:
                allstates.append(newobs)
                critics[option].add(critics[option].weights[phi])
                action_critics[option].add(action_critics[option].weights[phi])
                option_policies[option].add(option_policies[option].weights[phi])
                options_grid2obs[option][newobs] = len(critics[option].weights) -1



            next_observation, reward, done, _ = env.step(action)
            real_reward = reward
            rewards.append(reward)
            if observation != next_observation:
                G.add_edge(observation,next_observation)
            observation=next_observation
            observations.add(observation)
            epoch_states.add(observation)
            if reward and env.grid2obs.get(next_observation) != env.goal:
                reward_states.add(observation)
            

            pos[observation] += 0.1    
            if option ==1 or observation in map(env.obs2grid.get,env.init_states):
                optionsteps += 1

            if observation in allstates:
                next_option = 1
            else:
                if init_set:pdb.set_trace()
                next_option = 0  


            #### Updates
            phi = options_grid2obs[option].get(observation)
            critics[option].update(phi, reward, done)
            action_critics[option].update(phi, reward, done, critics[option].value(phi))

            critic_feedback = reward + (1.-done) * FLAGS.discount * critics[option].value(phi)
            critic_feedback -= critics[option].last_value

            option_policies[option].update(critic_feedback)


            action = option_policies[option].sample(phi,step) if np.random.rand() > 0.1 else np.random.randint(4)

            critics[option].save(phi)
            action_critics[option].save(phi, action)
            option_policies[option].save(phi,action)



            cumreward += real_reward
            last_phi = phi
            if done:
                break
        cumsteps += step
        print('Episode {} steps {} cumreward {} cumsteps {} optionsteps {}'.format(episode, step, cumreward, cumsteps,optionsteps))


        # Plot the V function
        # critic_map = env.grid.flatten().astype(float)
        # critic_vals = critics[option].weights.copy()
        # critic_vals[critic_vals == 0] = -0.02
        # critic_map[critic_map == 1] = -.02
        # critic_map[critic_map == 0] = critic_vals

        # pdb.set_trace()
        if FLAGS.path:
            # Plot the path
            path = np.ones((row*col))*0.25
            path[env.possible_states] = 0.      
            path[list(epoch_states)] = .5
            path[list(map(env.obs2grid.get,env.more_rewards_states))] = .6   
            path[env.obs2grid.get(env.goal)] = .7 
            path=path.reshape(row,col)


            fig,ax = plt.subplots(1,1)
            ax.imshow(path)
            # ax[1].imshow(path)
            directory = "path_gen{}_{}/".format(FLAGS.gen,row*col)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig("{}seed{}_{}.png".format(directory,seed,episode))
            plt.close();plt.clf()


        if done and not init_set:

            # pdb.set_trace()
            allobs = list(observations)
            allobs.sort()

            if env.obs2grid.get(env.goal) not in allobs:
                print("No sink this time")
                continue





            # Functions used to map rewards to probability distribution
            new_obs2grid = dict(zip( allobs, range(len(allobs)) )) 
            interpol =  make_interpolater(min(rewards), max(rewards), 0., 1.)
            def sigmoid(x, derivative=False):
              return x*(1-x) if derivative else 1/(1+np.exp(-x)) 

            # Set up labels
            source = new_obs2grid.get(sources[0])
            sink = new_obs2grid.get(env.obs2grid.get(env.goal))
            mapped_reward_states = map(new_obs2grid.get,list(reward_states))

            labels=np.zeros((len(allobs),2))
            labels[source] = [1. - sigmoid(0.), sigmoid(0.)]
            labels[sink] = [1. - sigmoid(1.), sigmoid(1.)]
            for r, m_r in zip(reward_states,mapped_reward_states):
                rew = env.original_rewards.get(env.grid2obs.get(r))
                labels[m_r] =  [1. - sigmoid(rew), sigmoid(rew)]

    
            # Set up edges and adjacency matrix
            edges = np.array(G.edges())
            graphsize = reduce(lambda x,y: x*y,edges.shape)
            edges = np.array( map(new_obs2grid.get,edges.reshape(graphsize,))).reshape(edges.shape)
            adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(allobs), len(allobs)), dtype=np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) # build symmetric adjacency matrix



            if want_graph:

                sess = tf.Session()


                init_set, V_weights = get_graph(sess, seed, edges, allobs, adj,
                                                labels, source, sink, mapped_reward_states, env)


                allstates=allobs
                nstates = len(allobs)


                option_policies.append(SoftmaxPolicy(rng, nstates, nactions, FLAGS.temperature) )
                critics.append(StateValue(FLAGS.discount, FLAGS.lr_critic, V_weights,) )
                action_critics.append(ActionValue(FLAGS.discount, FLAGS.lr_critic, np.zeros((nstates,nactions)),) )
                options_grid2obs.append(new_obs2grid)



            # break

        totalsteps.append(cumsteps)



        # Saving results
        # if want_graph and episode%100==0:

        #     np.savetxt("res/{}_graph_seed{}.csv".format(row*col,seed),totalsteps,delimiter=',')       
        # elif episode%100==0:

        #     np.savetxt("res/{}_nograph_seed{}.csv".format(row*col,seed),totalsteps,delimiter=',')       


