import numpy as np
import random
import math
import time

from collections import deque

import matplotlib
import matplotlib.pyplot

import tensorflow as tf
import gym

import cv2 as cv

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
try:
  tf.config.experimental.set_memory_growth(gpus[0], True)
except:
  print("Set error")

envName = "SpaceInvaders-v0"

env = gym.make(envName)
state = env.reset()

def preprocess(frame):
    img = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    img = cv.resize(img, (80, 105))
    return tf.expand_dims(img.astype(np.float32), axis=-1)/(255.0)

class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(4, (4,4), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(4, (4,4), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPool2D()
        self.flat1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation="relu")
        self.drop1 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(6)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat1(x)
        x = self.dense1(x)
        x = self.drop1(x)
        return self.dense2(x)

mainNet = DQN()
targetNet = DQN()

optimizer = tf.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()

#tf.keras.utils.plot_model(mainNet)
mainNet.build((None, 105, 80, 1))

#tf.keras.utils.plot_model(targetNet)

class ReplayBuffer(object):
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    
    def add(self, state, action, reward, nextState, done):
        self.buffer.append((state, action, reward, nextState, done))
    
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, n):
        states, actions, rewards, nextStates, dones = [], [], [], [], []
        ind = np.random.choice(len(self.buffer), n)
        for i in ind:
            elem = self.buffer[i]
            state, action, reward, nextState, done = elem
            states.append(np.array(state,copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            nextStates.append(np.array(nextState, copy=False))
            dones.append(done)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        nextStates = np.array(nextStates)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, nextStates, dones

def selectAction(state, epsilon):
    r = tf.random.uniform((1,))
    if(r < epsilon):
        return env.action_space.sample()
    else:
        return tf.argmax(mainNet(state)[0]).numpy()

@tf.function
def trainStep(states, actions, rewards, nextStates, dones):
    
    nextQs = targetNet(nextStates)
    maxNextQs = tf.reduce_max(nextQs, axis=-1)
    target = rewards + (1. - dones) * discount * maxNextQs
    
    with tf.GradientTape() as tape:
        qs = mainNet(states)
        actionsMask = tf.one_hot(actions, 6)
        maskedQs = tf.reduce_sum(actionsMask * qs, axis=-1)
        loss = mse(target, maskedQs)
        
    grads = tape.gradient(loss, mainNet.trainable_variables)
    optimizer.apply_gradients(zip(grads, mainNet.trainable_variables))

    return loss

numEpisode = 250
epsilon = 0.9
batchSize = 16
discount = 0.99
buffer = ReplayBuffer(100000)
curFrame = 0
epRecord = []

c = 0
env = gym.make(envName)

for i in range(numEpisode):
    state = env.reset()
    state = preprocess(env.render(mode='rgb_array'))
    epReward, done = 0.0, False
    
    while(not done):
        stateIn = tf.expand_dims(state, axis=0)
        action = selectAction(stateIn, epsilon)
        nextState, reward, done, info = env.step(action)
        nextState = preprocess(env.render(mode='rgb_array'))
        epReward += reward
        
        buffer.add(state, action, reward, nextState, done)
        state = nextState
        
        curFrame += 1
        
        if(curFrame % 5000 == 0):
            targetNet.set_weights(mainNet.get_weights())
        
        if(len(buffer) >= batchSize):
            states, actions, rewards, nextStates, dones = buffer.sample(batchSize)
            loss = trainStep(states, actions, rewards, nextStates, dones)

    if((i+1) % 3):
        if(epsilon > 0.05):
            epsilon -= 0.0025
    
    if(len(epRecord) == 50):
        epRecord = epRecord[1:]
    epRecord.append(epReward)
    
    if((i+1) % 5 == 0):
        print(f"Episode: {i+1}/{numEpisode}  Epsilon: {epsilon:.3f} 50 Last Reward : {np.mean(epRecord):.3f}")

    if((i+1)%10 == 0):
        mainNet.save_weights(f"weightMMM\weight{c}")
        targetNet.save_weights(f"weightTTT\weight{c}")

        print(f"Save Weight {c}")
        c += 1

env.reset()
done = 0
totalRew = 0
state = preprocess(env.render(mode='rgb_array'))
while(not done):
    state = tf.expand_dims(state, axis=0)
    action = selectAction(state, 0.01)
    sate, rew, done, info = env.step(action)
    env.render()
    totalRew += rew
    state = preprocess(state)

env.reset()
env.close()