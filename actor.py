# Policy Gradient Implementation
# REINFORCE
# Slides from Pieter Abbeel

import tensorflow as tf

#Define Actor Class
class Actor:

	#Instantiate with state and sction dimensions
	def __init__(dim_state, dim_action):
		self.dim_state = dim_state
		self.dim_action = dim_action

	#Creates Neural Network
	def createModel():
		x = tf.placeholder(tf.float32, [None, self.dim_state])
		
		pass

	#Get action for a given state
	def act(state):
		"""
		@params : state
		returns : probability of each action
		"""
		pass

	# Get Gradients for log probabilties
	def gradient(state, action):
		"""
		get gradients for taken action and state
		"""
		pass

	#Train to update weights
	def train():
		"""
		Trains neural network. Update parameters
		"""
		pass



