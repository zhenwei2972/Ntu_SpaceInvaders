# Leon Gurtler
import gym, time, cv2
import numpy as np

from sklearn.tree import DecisionTreeClassifier

env = gym.make("SpaceInvaders-ram-v0")
env.frameskip = 4
n = np.shape(env.observation_space.sample())
num_labels = env.action_space.n

# To save the training data
X_train = []
y_train = []

# variables and CONSTANTS
epsilon = .5
EPSILON_DECAY = .95
TOTAL_GAMES = 100_000

score_list = []
score_threshold = 150


# probably a good idea to initialize your model somewhere here
model = DecisionTreeClassifier()
trained = False


# training loop
for game_nr in range(TOTAL_GAMES):
    done = score = 0
    obs = env.reset()

    # initialize local game_based memory
    game_obs = []
    game_action = []

    while not done:
        if np.random.uniform() < epsilon or not trained: # basic epsilon-decreasing strategy
            action = env.action_space.sample()
        else:
            # this is where your action should be executed
            #action = env.action_space.sample() # please change this eh
            action=model.predict(np.asarray([obs])).astype(int)[0]
            #print(action)

        # append the observation, action pair to the local memory
        game_obs.append(obs)
        game_action.append(action)
        obs, reward, done, info = env.step(action)
        score+= reward

    score_list.append(score)

    if score >= score_threshold:
        for obs, a in zip(game_obs, game_action):
            y_train.append(a)
            X_train.append(obs)

    #if not (game_nr%50):
    print(f"{game_nr} / {TOTAL_GAMES}"+\
          f"\tMost recent score: {score}"+\
          f"\tInter-training score-avg: {np.mean(score_list)}", end="\r")


    #    Train the Network/tree on some condition
    if (not (game_nr+1)%200) and not (len(y_train) == 0):
        #print(np.shape(X_train))
        #print(np.shape(y_train))
        model.fit(
            np.asarray(X_train),
            np.asarray(y_train)
        )
        print(f"\nCurrent tree depth: {model.tree_.max_depth}\ts_th: {score_threshold}")
        print(score)
        X_train = []
        y_train = []
        trained=True
        epsilon *= EPSILON_DECAY

        score_threshold = np.max([np.percentile(score_list, 50), score_threshold])
        score_list = []

import pickle
filename = 'tree.sav'
pickle.dump(model, open(filename, 'wb'))

for _ in range(25):
    done = score = 0
    obs = env.reset()
    while not done:
        action = np.argmax(model.predict(np.asarray([obs])))
        obs, reward, done, info = env.step(action)
        env.render()    # actually render the games
        score += reward
        print(score, end="\r")
    print("") # necessary since we use end="\r" above but don't want to overwrite