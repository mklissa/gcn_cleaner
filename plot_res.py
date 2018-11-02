import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import numpy as np
import pdb
import matplotlib
sns.set(style='ticks')

# row=29
# col=44

size=1320

bias=140
data=[]
axes=[]

for seed in range(10):
	if seed==4:continue
	if seed==3:continue
	# if seed==0:continue
	# if seed==1:continue
	seed+=bias
	print(seed)
	mydat = np.genfromtxt('res/{}_graph_seed{}.csv'.format(size,seed), delimiter=',')
	print(len(mydat))
	data.append(mydat)
# pdb.set_trace()
axes.append(sns.tsplot(data=data,legend=True,condition='Diffusion-Based Approximate VF',color='red'))



data=[]
for seed in range(10):
	if seed==2:continue
	seed+=bias
	print(seed)
	# data.append(np.array(mydat)+np.random.randint(0,10000))
	data.append(np.genfromtxt('res/{}_nograph_seed{}.csv'.format(size,seed), delimiter=','))
axes.append(sns.tsplot(data=data,legend=True,condition='Actor-Critic',color='blue'))


plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Episodes',fontsize=24)
plt.ylabel('Cumulative Steps',fontsize=24)
plt.legend(loc=2,prop={'size': 16})

# plt.title('FourRooms Domain (29x29)',fontsize=24)
plt.title('Maze Domain',fontsize=24)
plt.tight_layout()
plt.savefig('fourroom_{}_results.png'.format(size))
plt.clf()
