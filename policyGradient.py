#main file
#Policy Gradient: using REINFORCE Algorithm
import gym
import critic

if __name__=="__main__":
	#initialize gym environment
	env = gym.make("CartPole-v0")
	observation = env.reset()
	#for n policies 	
	for i in range(1):
		#for each rollout
		T={}
		for _ in range(1000):
			env.render()
			action = env.action_space.sample() # your agent here (this takes random actions)
			observation, reward, done, info = env.step(action)
			T={T,(observation,reward,action)}
			if done:
				break
		print T
		
