#main file
#Policy Gradient: using REINFORCE Algorithm
import gym
import critic
import actor

import numpy as np
NUM_EPOCHS = 1
GAMMA = 0.99

if __name__=="__main__":
	#initialize gym environment
	env = gym.make("CartPole-v0")
	observation = env.reset()
	print observation.shape

	#Initialize Actor
	actor = actor.Actor(4,2)
	actor.createModel()	
	#Initialize Critic
	critic = critic.Critic(4)
	critic.createModel()	

	
	#for n policies 		
	for i in range(NUM_EPOCHS):
		#for each rollout
		T=[]
		for _ in range(100):
			env.render()
			observation = np.reshape(observation,(1,4))
			action_prob = actor.act(observation)
			action = np.argmax(action_prob)
			observation, reward, done, info = env.step(action)
			observation = np.reshape(observation,(1,4))
			T.append([observation,action,reward])
			if done:
				break
#		print T
		
		#Got trajectory
		# Get Rt
	
		T[len(T)-1].append(T[len(T)-1][2])		
		for i in range(len(T)-2, -1, -1):
			T[i].append(T[i][2]+GAMMA*T[i+1][3])
#		print T		
		# find bt which is value
		for i in range(len(T)):
			T[i].append(critic.Value(T[i][0]))
		#find A which is Advantage (R-b)
		for i in range(len(T)):
			T[i].append(T[i][-2]-T[i][-1])
		print T[1]
	

		
