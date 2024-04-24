import gym
import numpy as np
import os
import torch
import argparse

from logger import Logger
from utils import stack_frames, add_distractor


parser = argparse.ArgumentParser()

parser.add_argument('--env-name', type=str, default='Pendulum-v1',
                    help='Environment name.')
parser.add_argument('--observation-dim-w', type=int, default=84,
                    help='Width of the input measurements (RGB images).')
parser.add_argument('--observation-dim-h', type=int, default=84,
                    help='Height of the input measurements (RGB images).')
parser.add_argument('--test', default=True,
                    help='Generate training or testing dataset.')
parser.add_argument('--training-dataset', type=str, default='pendulum-train-moving.pkl',
                    help='Training dataset.')
parser.add_argument('--testing-dataset', type=str, default='pendulum-test-moving.pkl',
                    help='Testing dataset.')
parser.add_argument('--random-policy', default=True,
                    help='Use random action policy.')
parser.add_argument('--render-mode', type=str, default='rgb_array',
                    help='Render mode (human or rgb_array)')
parser.add_argument('--distractors', type=str, default='moving',
                    help='Distractors type (none, fixed, moving)')

args = parser.parse_args()

env_name = args.env_name
test = args.test
if test:
    seed = 7
    num_episodes = 5
    data_file_name = args.testing_dataset
else:
    num_episodes = 50
    seed = 1
    data_file_name = args.training_dataset

obs_dim1 = args.observation_dim_w
obs_dim2 = args.observation_dim_h
random_policy = args.random_policy
distractors = args.distractors

env = gym.make(env_name, render_mode=args.render_mode)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

directory = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(directory + '/data/')
logger = Logger(folder)

# Set seeds
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

max_steps = 200 #default for pendulum-v1

for episode in range(num_episodes):
    state, _ = env.reset(seed=seed)
    frame = np.array(env.render())
    prev_frame = np.array(env.render())
    print('Episode: ', episode)
    for step in range(max_steps):
        if distractors == 'fixed':
            frame = add_distractor(frame, is_random=False)
        if distractors == 'moving':
            frame = add_distractor(frame, is_random=True)
        obs = stack_frames(prev_frame, frame, obs_dim1, obs_dim2)
        if random_policy:
            action = env.action_space.sample()
        else:
            pass
        next_state, reward, done, truncated, info = env.step(action)

        next_frame = np.array(env.render())
        next_obs = stack_frames(frame, next_frame, obs_dim1, obs_dim2)

        if step == max_steps - 1:
            done = True

        logger.obslog((obs, action, reward, next_obs, done, state))

        prev_frame = frame
        frame = next_frame
        state = next_state

        if done:
            break

logger.save_obslog(filename=data_file_name)