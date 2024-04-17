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


    def sample_batch(self, batch_size=32 , distractor=False, fixed=True, dist_size=20):
        idxs = np.random.randint(0, self.size, size=batch_size)
        idxs2 = np.random.randint(0, self.size, size=batch_size)

        for i in range(idxs.shape[0]):
            if self.done_buf[idxs[i]] == 1.0:
                idxs[i] = idxs[i] - 1

        if distractor:
            if fixed:
                distractor = np.zeros((dist_size, dist_size, self.obs_dim[2]))
                x_dist = 10
                y_dist = 11
                obs1_temp = np.copy(self.obs_buf[idxs])
                next_obs1_temp = np.copy(self.next_obs_buf[idxs])
                obs2_temp = np.copy(self.obs_buf[idxs2])
                next_obs2_temp = np.copy(self.next_obs_buf[idxs2])
            else:
                distractor = np.zeros((dist_size, dist_size, self.obs_dim[2]))
                x_dist = np.random.randint(0, self.obs_dim[0] - dist_size, size=batch_size)
                y_dist = np.random.randint(0, self.obs_dim[1] - dist_size, size=batch_size)
                obs1_temp = np.copy(self.obs_buf[idxs])
                next_obs1_temp = np.copy(self.next_obs_buf[idxs])
                obs2_temp = np.copy(self.obs_buf[idxs2])
                next_obs2_temp = np.copy(self.next_obs_buf[idxs2])

            obs1_temp, next1_obs_temp, obs2_temp, next2_obs_temp = self.add_distractor(distractor, x_dist, y_dist,
                                                                                       obs1_temp, next_obs1_temp,
                                                                                       obs2_temp, next_obs2_temp, fixed)
            return dict(obs1=obs1_temp,
                        obs2=next1_obs_temp,
                        acts=self.acts_buf[idxs],
                        rews=self.rews_buf[idxs],
                        done=self.done_buf[idxs],
                        ep_start=self.ep_start_buf[idxs],
                        obs3=next_obs2_temp,
                        obs4=obs2_temp,
                        acts2=self.acts_buf[idxs2],
                        rews2=self.rews_buf[idxs2],
                        states=self.state_buf[idxs],
                        next_states=self.state_buf[idxs + 1])

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

    def add_distractor(self, distractor, x_dist, y_dist, obs1_temp, next_obs1_temp, obs2_temp, next_obs2_temp,
                       fixed=False):
        if fixed:
            for n in range(obs1_temp.shape[0]):
                obs1_temp[n][x_dist:x_dist+distractor.shape[0], y_dist:y_dist+distractor.shape[1], :] = distractor[:,:,:]
                next_obs1_temp[n][x_dist:x_dist+distractor.shape[0], y_dist:y_dist+distractor.shape[1], :] = distractor[:,:,:]
                obs2_temp[n][x_dist:x_dist+distractor.shape[0], y_dist:y_dist+distractor.shape[1], :] = distractor[:,:,:]
                next_obs2_temp[n][x_dist:x_dist+distractor.shape[0], y_dist:y_dist+distractor.shape[1], :] = distractor[:,:,:]
        else:
            for n in range(obs1_temp.shape[0]):
                obs1_temp[n][x_dist[n]:x_dist[n] + distractor.shape[0], y_dist[n]:y_dist[n] + distractor.shape[1], :] = distractor[:, :, :]
                next_obs1_temp[n][x_dist[n]:x_dist[n] + distractor.shape[0], y_dist[n]:y_dist[n] + distractor.shape[1], :] = distractor[:, :, :]
                obs2_temp[n][x_dist[n]:x_dist[n] + distractor.shape[0], y_dist[n]:y_dist[n] + distractor.shape[1], :] = distractor[:, :, :]
                next_obs2_temp[n][x_dist[n]:x_dist[n] + distractor.shape[0], y_dist[n]:y_dist[n] + distractor.shape[1], :] = distractor[:, :, :]


        obs1_temp = (obs1_temp).clip(0.0, 1.0)
        next_obs1_temp = (next_obs1_temp).clip(0.0, 1.0)
        obs2_temp = (obs2_temp).clip(0.0, 1.0)
        next_obs2_temp = (next_obs2_temp).clip(0.0, 1.0)

        return obs1_temp, next_obs1_temp, obs2_temp, next_obs2_temp

    def get_all_samples(self):
        return dict(obs=self.obs_buf[:self.size],
                    next_obs=self.next_obs_buf[:self.size],
                    acts=self.acts_buf[:self.size],
                    done=self.done_buf[:self.size],
                    states=self.state_buf[:self.size])
    def clear_memory(self):
        self.__init__(self.obs_dim, self.act_dim, self.max_size, self.state_dim)































