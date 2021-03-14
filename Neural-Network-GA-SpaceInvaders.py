# Leon Gurtler
import gym, time, cv2, random
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten

env = gym.make("SpaceInvaders-ram-v0")
env.frameskip = 4
n = np.shape(env.observation_space.sample())
num_labels = env.action_space.n

score_list = []

# GA CONSTANTS
GENERATION_SIZE = 128
EPOCHS = 10
KEEP_BEST_N = 12
ASSESS_N_GAMES = 3

MUTATION_RATE = .7
MUTATION_PROB = .7


# create the Neural Network Architecture
model = Sequential()

model.add(Dense(64, activation="tanh", input_shape=(n)))
model.add(Dense(32, activation="tanh"))
model.add(Dense(num_labels))

shapes = [weight.shape for weight in model.get_weights()]
shape = (1,) + env.observation_space.shape

# functions and procedures for the GA
def get_solution(weights):
  return np.concatenate([weight.reshape(-1) for weight in weights])

def set_weights(solution):
  model.set_weights([solution[1:1+np.prod(shape)].reshape(shape) for shape in shapes])

def get_action(observation):
  return np.argmax(model.predict_on_batch(observation))

def evaluate_fitness(gene):
    """
    Evalute the average score achieved by the currently tested gene on a fixed
    number of attempts. The average might be useful to filter some noise.
    """
    global ASSESS_N_GAMES
    set_weights(gene)
    total_score = 0
    for _ in range(ASSESS_N_GAMES):
        done = 0 # in python 0 == False
        obs = env.reset()
        while not done:
            # choose the action with the highest expected score
            action = get_action(np.asarray([obs/255.0]))
            obs, reward, done, info = env.step(action)
            total_score += reward
    return total_score/ASSESS_N_GAMES

def initial_population(raw_gene):
    """
    Create multiple similar but inharently random weights for the first epoch.
    """
    global GENERATION_SIZE
    theta_flat_shape = np.shape(raw_gene)
    new_generation_shape = (GENERATION_SIZE, *list(theta_flat_shape))
    new_generation = np.random.uniform(-1, 1, size=(new_generation_shape))

    return new_generation

def repopulate(fittest_n_genes):
    """
    After the performance of all genes has been assessed, we take the best n genes
    and use them to re-populate our generation.
    """
    global MUTATION_RATE, MUTATION_PROB, GENERATION_SIZE
    theta_flat_shape = np.shape(fittest_n_genes[0])
    new_generation_shape = (GENERATION_SIZE, *list(theta_flat_shape))
    new_generation = np.zeros((new_generation_shape))

    # keep the fittest_n_genes in the next generation (for robustness)
    new_generation[:len(fittest_n_genes)] = fittest_n_genes.copy()

    for x in range(len(fittest_n_genes), GENERATION_SIZE):
        parent_mask = np.random.uniform(size=theta_flat_shape)>.5
        mutation_mask = np.random.uniform(1-MUTATION_RATE, 1+MUTATION_RATE, size=(theta_flat_shape)) * \
                        (np.random.uniform(size=(theta_flat_shape))<MUTATION_PROB)

        new_generation[x] = (random.choice(fittest_n_genes) * parent_mask + \
                            random.choice(fittest_n_genes) * (~parent_mask)) * \
                            mutation_mask
    return new_generation


# initialize first batch
genes_theta_matrix = initial_population(get_solution(model.get_weights()))

for epoch in range(EPOCHS):
    assessment_vector = np.zeros((len(genes_theta_matrix)))

    print(len(genes_theta_matrix))

    for x in range(len(genes_theta_matrix)):
        assessment_vector[x] = evaluate_fitness(gene=genes_theta_matrix[x])
        print(f"assessing: {x}\tscore: {assessment_vector[x]}", end="\r")

    print(f"{epoch} / {EPOCHS}\t"+\
          f"\tepoch_max: {np.max(assessment_vector):.0f}\t"+\
          f"\tepoch_mean: {np.mean(assessment_vector):.0f}\t"+\
          f"\tgames_played: {epoch*GENERATION_SIZE*ASSESS_N_GAMES}")#, end="\r")


    # pick the cut-off point for score and pass all genes with a greater or equal
    # score into the re-population function.
    best_nth_value = np.sort(assessment_vector)[-KEEP_BEST_N]
    genes_theta_matrix = repopulate(
        fittest_n_genes=genes_theta_matrix[np.where(assessment_vector>=best_nth_value)[:KEEP_BEST_N]]
        )

    model.save("myAwesomeModelLah.model")




# choose the best gene of the last epoch to render a few games
best_gene = genes_theta_matrix[np.argmax(assessment_vector)]
set_weights(best_gene)
for _ in range(25):
    done = score = 0
    obs = env.reset()
    while not done:
        action = get_action(np.asarray([obs]))
        obs, reward, done, info = env.step(action)
        env.render()    # actually render the games
        score += reward
        print(score, end="\r")
    print("") # necessary since we use end="\r" above but don't want to overwrite
