import numpy as np

import gym

import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras

import cv2 as cv

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
try:
  tf.config.experimental.set_memory_growth(gpus[0], True)
except:
  print("Set error")

env = gym.make("SpaceInvaders-v0")

done = reward = 0
env.reset()

frame = 0

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

#14 good
c = 14

mainNet = DQN()
mainNet.load_weights(f"weightMMM\weight{c}")

def selectAction(state, epsilon):
    r = tf.random.uniform((1,))
    if(r < epsilon):
        return env.action_space.sample()
    else:
        out = mainNet(state)
        print(out)
        return tf.argmax(out[0]).numpy()

done = 0
totalRew = 0
state = preprocess(env.render(mode='rgb_array'))
while(not done):
    state = tf.expand_dims(state, axis=0)
    action = selectAction(state, 0.01)
    sate, rew, done, info = env.step(action)
    env.render()
    totalRew += rew
    frame += 1
    state = preprocess(env.render(mode='rgb_array'))

print(f"{c} - Reward: {totalRew} Frame: {frame}")

env.reset()
env.close()