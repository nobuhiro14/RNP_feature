import torch
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    def __init__(self,size,frame_history_len,acts_per_episode):


        self.size = size
        self.frame_history_len = frame_history_len
        self.acts_per_episode = acts_per_episode

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.xlen     = None 
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def sample(self,batch_size):
        """Sample `batch_size` different transitions.
        i-th sample transition is the following:
        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: torch.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype torch.uint8
        act_batch: torch.array
            Array of shape (batch_size,) and dtype torch.int32
        rew_batch: torch.array
            Array of shape (batch_size,) and dtype torch.float32
        next_obs_batch: torch.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype torch.uint8
        done_mask: torch.array
            Array of shape (batch_size,) and dtype torch.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def _encode_sample(self,idxes):
        #obs_batch       = torch.cat([self._encode_observation(idx)[None] for idx in idxes] ,0)
        obs_batch       = [self.obs[ids] for ids in idxes]
        xlen_batch      = [obs_batch[i].shape[0] for i in range(len(obs_batch))]
        obs_batch       = pad_sequence(obs_batch,batch_first=True,padding_value = 0)
        act_batch       = self.action[idxes]
        rew_batch       = self.reward[idxes]
        nx_idxes = list(np.array(idxes) + 1)

        #next_obs_batch  = torch.cat([self._encode_observation(idx+1)[None] for idx in idxes],0)
        next_obs_batch  = [self.obs[ids] for ids in nx_idxes]
        next_xlen_batch = [next_obs_batch[i].shape[0] for i in range(len(next_obs_batch))]
        next_obs_batch  = pad_sequence(next_obs_batch,batch_first=True,padding_value = 0)
        done_mask       = self.done[idxes]

        return obs_batch, xlen_batch, act_batch, rew_batch, next_obs_batch, next_xlen_batch,done_mask

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.
        Returns
        -------
        observation: torch.array
            Array of shape (img_c * frame_history_len, img_h, img_w)
            and dtype torch.uint8, where observation[i*img_c:(i+1)*img_c, :, :]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self,idx):
        end_idx = idx+1
        start_idx = end_idx - self.frame_history_len

        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        #if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size :
            start_idx = 0

        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]==1:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx-start_idx)

        if start_idx < 0 or missing_context > 0:
            frames = [torch.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return torch.cat(frames, 0) # c, h, w instead of h, w c
        else:
            # this optimization has potential to saves about 30% compute time \o/
            # c, h, w instead of h, w c
            sz = self.obs[0].shape[2]
            return self.obs[start_idx:end_idx].reshape(-1,sz)



    def store_frame(self, frame,action,reward,done):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.
        Parameters
        ----------
        frame: torch.array
            Array of shape (img_h, img_w, img_c) and dtype torch.uint8
            the frame to be stored
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """

        if self.obs is None:
            self.obs      = [[] for i in range(self.size)]
            self.action   = torch.empty([self.size],                     dtype=torch.float32)
            self.reward   = torch.empty([self.size],                     dtype=torch.float32)
            self.done     = torch.empty([self.size],                     dtype=torch.int32)
            print("create box")
        ret = self.next_idx

        for i in range(self.acts_per_episode):
            tmp = (ret+i) % self.size
            self.obs[tmp] = frame[i][0]
            self.action[tmp] = action[i]
            self.reward[tmp] = reward[i]
            self.done[tmp] = done[i]



        self.next_idx = (self.next_idx +self.acts_per_episode) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + self.acts_per_episode)
