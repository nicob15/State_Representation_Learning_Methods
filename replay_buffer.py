import numpy as np

class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size, state_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.size = size
        self.obs_buf = np.zeros([int(size), int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2])], dtype=np.float32)
        self.next_obs_buf = np.zeros([int(size), int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2])], dtype=np.float32)
        self.acts_buf = np.zeros([int(size), int(act_dim)], dtype=np.float32)
        self.rews_buf = np.zeros(int(size), dtype=np.float32)
        self.done_buf = np.zeros(int(size), dtype=np.float32)
        self.ep_start_buf = np.zeros(int(size), dtype=bool)
        self.state_buf = np.zeros([int(size), state_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, int(size)


    def store(self, obs, act, rew, next_obs, done, state):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.state_buf[self.ptr] = state
        self.ptr = (self.ptr + 1) % self.max_size  # replace oldest entry from memory
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        idxs2 = np.random.randint(0, self.size, size=batch_size)

        for i in range(idxs.shape[0]):
            if self.done_buf[idxs[i]] == 1.0:
                idxs[i] = idxs[i] - 1

        return dict(obs1=self.obs_buf[idxs],
                    obs2=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    ep_start=self.ep_start_buf[idxs],
                    obs3=self.next_obs_buf[idxs2],
                    obs4=self.obs_buf[idxs2],
                    acts2=self.acts_buf[idxs2],
                    rews2=self.rews_buf[idxs2],
                    states=self.state_buf[idxs],
                    next_states=self.state_buf[idxs+1])

    def sample_sequence(self, start_idx=0, seq_len=5):
        end_idx = start_idx + seq_len
        return dict(obs1=self.obs_buf[np.arange(start_idx, end_idx)],
                    acts=self.acts_buf[np.arange(start_idx, end_idx)],
                    obs2=self.next_obs_buf[np.arange(start_idx, end_idx)],
                    states=self.state_buf[np.arange(start_idx, end_idx)],
                    next_states=self.state_buf[np.arange(start_idx+1, end_idx+1)],
                    )

    def get_all_samples(self):
        return dict(obs=self.obs_buf[:self.size],
                    next_obs=self.next_obs_buf[:self.size],
                    acts=self.acts_buf[:self.size],
                    done=self.done_buf[:self.size])

    def clear_memory(self):
        self.__init__(self.obs_dim, self.act_dim, self.max_size, self.state_dim)































class ReplayBuffer_v2:
    "see: https://spinningup.openai.com/en/latest/_modules/spinup/algos/ddpg/ddpg.html#ddpg"
    "Possible extensions (CER): https://arxiv.org/pdf/1712.01275.pdf"
    " (PER): https://arxiv.org/abs/1511.05952"
    CER = False


    # TODO: Extend with possibility of restoring sequences
    def __init__(self, obs_dim, act_dim,  size, state_dim, random_sampling=True, frame_stacked=2):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.size = size
        self.random_sampling = random_sampling
        self.obs_buf = np.zeros([int(size), int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2])], dtype=np.float32)
        self.next_obs_buf = np.zeros([int(size), int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2])], dtype=np.float32)
        self.acts_buf = np.zeros([int(size), int(act_dim)], dtype=np.float32)
        self.rews_buf = np.zeros(int(size), dtype=np.float32)
        self.done_buf = np.zeros(int(size), dtype=np.float32)
        self.ep_start_buf = np.zeros(int(size), dtype=bool)
        self.state_buf = np.zeros([int(size), state_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, int(size)
        self.frame_stacked = frame_stacked


    def store(self, obs, act, rew, next_obs, done, state):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.state_buf[self.ptr] = state
        self.ptr = (self.ptr + 1) % self.max_size  # replace oldest entry from memory
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self, batch_size=32):


        idxs = np.random.randint(0, self.size, size=batch_size)
        idxs2 = np.random.randint(0, self.size, size=batch_size)

        for i in range(idxs.shape[0]):
            if self.done_buf[idxs[i]] == 1.0:
                idxs[i] = idxs[i] - 1
            for j in range(self.frame_stacked):
                if self.done_buf[idxs[i] - j] == 1.0:
                    idxs[i] = idxs[i] - 1

        noisy_obs_buff = np.array((batch_size, self.state_dim))
        noisy_next_obs_buff = np.array((batch_size, self.state_dim))

        for i in range(batch_size):
            noisy_obs_buff[i] = add_gaussian_noise(self.obs_buf[idxs[i]], noise_level=0.5, clip=True)
            noisy_next_obs_buff[i] = add_gaussian_noise(self.next_obs_buf[idxs[i]], noise_level=0.5, clip=True)

        if self.CER: idxs[-1] = abs(self.ptr - 1) # this takes the last added sample, unless ptr = 0, then it takes sample 1, this then does violate CER

        return dict(obs1=noisy_obs_buff,
                    obs2=noisy_next_obs_buff,
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    ep_start=self.ep_start_buf[idxs],
                    obs3=self.next_obs_buf[idxs2],
                    states=self.state_buf[idxs])

    def sample_sequence(self, start_idx=0, seq_len=5):
        return dict(obs1=self.obs_buf[np.arange(start_idx, seq_len)])


    def get_all_samples(self):
        return dict(obs=self.obs_buf[:self.size],
                    next_obs=self.next_obs_buf[:self.size],
                    acts=self.acts_buf[:self.size],
                    rews=self.rews_buf[:self.size],
                    done=self.done_buf[:self.size])
                    #sample_nr=self.sample_nr_buf[:self.size])

    def clear_memory(self):
        self.__init__(self.obs_dim, self.act_dim, self.max_size)