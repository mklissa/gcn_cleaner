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

# pdb.set_trace()
# colors = [(0,0,0)] +[(cm.viridis(i)) for i in xrange(1,256)]
# new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)
colors = [(0,0,0)] +[(0.5,0.5,0.5)]+ [(cm.viridis(i)) for i in xrange(2,256)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


# seeds = [149]
# seeds = range(140,150)
# want_graph=1
plotsteps=0



flags.DEFINE_integer('gen', 1, 'Do you want to generate a graph?')
flags.DEFINE_integer('seed', 0, 'Random seed.')
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




flags.DEFINE_integer('ngraph', 10, "Number of episodes before graph generation")
flags.DEFINE_integer('nepisodes', 500, "Number of episodes per run")
flags.DEFINE_integer('nruns',1, "Number of runs")
flags.DEFINE_integer('nsteps',1000, "Maximum number of steps per episode")
flags.DEFINE_integer('noptions',1, 'Number of options')
flags.DEFINE_integer('baseline',1, "Use the baseline for the intra-option gradient")
flags.DEFINE_integer('primitive',0, "Augment with primitive")



flags.DEFINE_float('temperature',1e-3, "Temperature parameter for softmax")
flags.DEFINE_float('discount',0.99, 'Discount factor')
flags.DEFINE_float('lr_intra',1e-1, "Intra-option gradient learning rate")
flags.DEFINE_float('lr_critic',1e-1, "Learning rate")





# flags.DEFINE_integer('hidden1', 500, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 250, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden4', 28, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden5', 18, 'Number of units in hidden layer 1.')



# flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden3', 46, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden4', 28, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden5', 18, 'Number of units in hidden layer 1.')


flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 46, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden5', 8, 'Number of units in hidden layer 1.')



# flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden4', 32, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden5', 16, 'Number of units in hidden layer 1.')

want_graph= FLAGS.gen
seeds = [FLAGS.seed]
totalsteps = []

for seed in seeds:
    print('seed:',seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    rng = np.random.RandomState(seed)

    
    env = Fourrooms()
    walls = np.argwhere(env.occupancy.flatten()==1)
    row,col = env.occupancy.shape
    # pdb.set_trace()
    features = Tabular(env.observation_space.n)
    nfeatures, nactions = len(features), env.action_space.n

    # pdb.set_trace()
    observations = set()
    G = nx.Graph()
    G.add_nodes_from(range(len(env.occupancy.flatten())))
    full_grid_dict = dict(zip(range(env.observation_space.n), 
                            np.argwhere(env.occupancy.flatten()==0).squeeze() ))

    # pdb.set_trace()

    options_dict = [dict(zip( np.argwhere(env.occupancy.flatten()==0).squeeze(),
                            range(env.observation_space.n) )), ]

    option_policies = [SoftmaxPolicy(rng, nfeatures, nactions, FLAGS.temperature,FLAGS.lr_intra), ]
    # intraoption_improvement = IntraOptionGradient(option_policies, FLAGS.lr_intra)

    critics = [StateValue(FLAGS.discount, FLAGS.lr_critic, np.zeros((nfeatures)) ), ]
    action_critics = [ActionValue(FLAGS.discount, FLAGS.lr_critic, np.zeros((nfeatures,nactions)) ), ]

    done=False
    cumsteps = 0.
    optionsteps = 0.
    myrand = 1.
    myrandinit =1.
    sources=[]
    init_set=[]
    goals = []
    allstates=[]
    for episode in range(FLAGS.nepisodes):
        epoch_states= set()
        phis = []
        rewards = []
        pos = env.occupancy.flatten().astype(float)
        pos[pos == 1] = -.5
        
        observation = env.reset()
        # start = full_grid_dict.get(observation)
        start=observation
        sources.append(start)
        observations.add(start)


        if init_set:
            option=1
        else:
            option=0

        newpos = options_dict[option].get(observation)
        newobs = observation
        if newpos is None and newobs not in walls:
            allstates.append(newobs)
            critics[option].add(np.zeros((1)) )
            action_critics[option].add(np.zeros((1,4)) )
            option_policies[option].add(np.zeros((1,4)) )
            options_dict[option][newobs] = len(critics[option].weights) -1



        last_phi = phi = options_dict[option].get(observation)
        phis.append(phi)

        action = option_policies[option].sample(phi,0)  if np.random.rand() > 0.1 else np.random.randint(4)
        critics[option].save(phi)
        action_critics[option].save(phi, action)
        option_policies[option].save(phi,action)

        
        cumreward = 0.
        for step in range(FLAGS.nsteps):



            acts = [-col,col,-1,1]
            newpos = options_dict[option].get(observation+acts[action])
            newobs = observation+acts[action]
            if newpos is None and newobs not in walls:
                allstates.append(newobs)
                critics[option].add(critics[option].weights[phi])
                action_critics[option].add(action_critics[option].weights[phi])
                option_policies[option].add(option_policies[option].weights[phi])
                options_dict[option][newobs] = len(critics[option].weights) -1



            next_observation, reward, done, _ = env.step(action)
            real_reward = reward
            rewards.append(reward)
            if observation != next_observation:
                G.add_edge(observation,next_observation)
            observation=next_observation
            observations.add(observation)
            epoch_states.add(observation)
            

            pos[observation] += 0.1    
            if option ==1 or observation in map(env.state_dict.get,env.init_states):
                optionsteps += 1

            if observation in allstates:
                next_option = 1
            else:
                if init_set:pdb.set_trace()
                next_option = 0  



            if (done or step==FLAGS.nsteps-1) and 1 and episode>=5:
            # if init_set and (done or step==FLAGS.nsteps-1) and plotsteps:
            # if init_set and step>0 and plotsteps:


                # pdb.set_trace()
                allobs = list(observations)
                allobs.sort()

                # # Plot the V function
                print(critic_vals.shape)
                critic_map = env.occupancy.flatten().astype(float)
                critic_vals = critics[option].weights.copy()
                critic_vals[critic_vals == 0] = -0.01
                critic_map[critic_map == 1] = -.02


                critic_map[options_dict[option].keys()] = critic_vals



                # #Plot the path
                path = env.occupancy.flatten().astype(float)
                walls = np.where(env.occupancy.flatten()==1)[0]
                # pdb.set_trace()
                # print(walls)
                path[walls]= 0.1            
                # path[path == 1] = .4
                path[allobs] = 0.4
                path[list(epoch_states)] = 0.7
                path[start] = 0.6
                path[list(epoch_states)[-1]] = 0.5
                path[full_grid_dict.get(env.goal)] = 0.75


                fig,ax = plt.subplots(1)
                # ax.imshow(path.reshape(env.occupancy.shape),cmap=new_map)
                ax.imshow(critic_map.reshape(env.occupancy.shape),cmap=new_map)
                # ax.imshow(feats_map.reshape(env.occupancy.shape),cmap=new_map)
                # plt.show()
                epoch_states=set()

                plt.xticks([])
                plt.yticks([])  
                
                directory = "presvf_gcn_output/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig("{}epoch{}_seed{}.png".format(directory,episode,seed))
                plt.close()


            phi = options_dict[option].get(observation)
            critics[option].update(phi, reward, done)
            action_critics[option].update(phi, reward, done, critics[option].value(phi))



            critic_feedback = reward + (1.-done) * FLAGS.discount * critics[option].value(phi)
            critic_feedback -= critics[option].last_value


            option_policies[option].update(critic_feedback)

            action = option_policies[option].sample(phi,step) if np.random.rand() > 0.1 else np.random.randint(4)
            phis.append(phi)




            critics[option].save(phi)
            action_critics[option].save(phi, action)
            option_policies[option].save(phi,action)




            cumreward += real_reward
            last_phi = phi
            if done:
                break
        cumsteps += step
        print('Episode {} steps {} cumreward {} cumsteps {} optionsteps {}'.format(episode, step, cumreward, cumsteps,optionsteps))






        
        # if episode == FLAGS.nepisodes-1:
        # if episode == FLAGS.ngraph -1:
        if done and not init_set:
        # if 0:

            # pdb.set_trace()
            allobs = list(observations)
            allobs.sort()

            if full_grid_dict.get(env.goal) not in allobs:
                print("No sink this time")
                continue


            # Plot the V function
            critic_map = env.occupancy.flatten().astype(float)
            critic_vals = critics[option].weights.copy()
            critic_vals[critic_vals == 0] = -0.02
            critic_map[critic_map == 1] = -.02
            critic_map[critic_map == 0] = critic_vals


            # #plot feats
            # interpol = make_interpolater(min(critics[option].weights),max(critics[option].weights),0.,1.)
            # xtra_feats =[]
            # for w in critics[option].weights:
            #     xtra_feats.append(interpol(w))
            # feats_map = env.occupancy.flatten().astype(float)
            # feats_vals = np.array(xtra_feats).copy()
            # feats_map[feats_map == 1] = -.02
            # feats_map[feats_map == 0] = feats_vals


            # #Plot the path
            # path = env.occupancy.flatten().astype(float)
            # walls = np.where(env.occupancy.flatten()==1)[0]
            # pdb.set_trace()
            # print(walls)
            # path[walls]= 0.1            
            # path[path == 1] = .4
            # path[allobs] = 0.4
            # path[list(epoch_states)] = 0.7
            # path[start] = 0.6
            # path[list(epoch_states)[-1]] = 0.5
            # path[full_grid_dict.get(env.goal)] = 0.75


            # fig,ax = plt.subplots(1)
            # ax.imshow(path.reshape(env.occupancy.shape),cmap=new_map)
          
            # ax.imshow(critic_map.reshape(env.occupancy.shape),cmap=new_map)
            # plt.show()
            # # ax[2].imshow(feats_map.reshape(env.occupancy.shape),cmap=new_map)
            # # plt.show()
            # epoch_states=set()

            # # # Plot the Q functions
            # # acts = ['up','down','left','right']
            # # for i,act in zip(range(env.action_space.n),acts):
                
            # #     plan = env.occupancy.flatten().astype(float)
            # #     plot_critic = action_critic.weights[:,0,i].copy()
            # #     plot_critic[plot_critic == 0] = -0.01
            # #     plan[plan == 0] = plot_critic
            # #     plan[plan == 1] = -.02   
            # #     ax[i+2].imshow(plan.reshape(env.occupancy.shape),cmap=new_map)
            # plt.xticks([])
            # plt.yticks([])  

            # directory = "presentation/"
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            # plt.savefig("{}epoch{}_seed{}.png".format(directory,episode,seed))
            # plt.close()


            # pdb.set_trace()
            option=0
            interpol = make_interpolater(min(critics[option].weights),max(critics[option].weights),0.,1.)
            critic_features =[]
            for w in critics[option].weights:
                critic_features.append(interpol(w))
            critic_features[env.goal] = 1. # little hack

            # pdb.set_trace()
            allfeats = np.zeros_like(env.occupancy).flatten().astype(float)
            feat_indices = (env.occupancy == 0).flatten()
            allfeats[feat_indices] = critic_features


            row,col = env.occupancy.shape

            # pdb.set_trace()
            # sources = [start]
            # sources = map(env.state_dict.get,env.init_states)

            # sources = [sources[0] for _ in range(5) ]
            # for counter,source in enumerate(sources):
            #     if source not in allobs:
            #         continue

            source = sources[0]
            sink = full_grid_dict.get(env.goal)
            title=FLAGS.app
            with open("gcn/data/{}_edges.txt".format(title),"w+") as f:
                for line in nx.generate_edgelist(G, data=False):
                    f.write(line+"\n")

            with open("gcn/data/{}_info.txt".format(title),"w+") as f:
                f.write("{} {}\n".format(row,col))
                f.write("{} {}\n".format(source,sink))
                for state,feat in zip(allobs,allfeats[allobs]):
                    f.write("{} {}\n".format(state,feat))
            last_obs = allobs






            if want_graph:
                # pdb.set_trace()
                sess = tf.Session()
                # init_set,goals,sinks,V_weights = get_graph(seed,sess,sources[:10],[sink],0)
                init_set,goals,sinks,V_weights = get_graph(seed,sess,[sources[0]],[sink],0)
                # sess = tf.Session()
                # init_set,goals,_ = get_graph(seed,sess,sources[:3],sinks,1)

                # pdb.set_trace()
                
                # def replace(x):
                #     if x < .3:
                #         return 0.
                #     else:
                #         return x
                # V_weights = map(replace,V_weights)

                allstates = init_set + goals
                allstates.sort()
                nstates = len(allstates)

                # pdb.set_trace()
                option_policies.append(SoftmaxPolicy(rng, nstates, nactions, FLAGS.temperature) )
                # intraoption_improvement.add_option(option_policies)

                critics.append(StateValue(FLAGS.discount, FLAGS.lr_critic, V_weights,) )
                action_critics.append(ActionValue(FLAGS.discount, FLAGS.lr_critic, np.zeros((nstates,nactions)),) )

                options_dict.append( dict(zip( allstates, range(nstates) )) )
                # FLAGS.nsteps = 10000


            # break

        totalsteps.append(cumsteps)
    # pdb.set_trace()


          
        # if want_graph and episode%100==0:

        #     np.savetxt("res/{}_graph_seed{}.csv".format(row*col,seed),totalsteps,delimiter=',')       
        # elif episode%100==0:

        #     np.savetxt("res/{}_nograph_seed{}.csv".format(row*col,seed),totalsteps,delimiter=',')       


