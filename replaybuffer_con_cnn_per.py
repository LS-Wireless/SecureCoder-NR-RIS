import torch
import numpy as np


class ReplayBuffer_cnn:
    def __init__(self, args, k=0.2, priority_ratio=0.3):
        """Initialize the priority replay buffer with main and priority storage"""
        # Device configuration (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Main buffer parameters
        self.batch_size = args.batch_size
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        # Main buffer arrays for storing transitions
        self.s = np.zeros((args.batch_size, 2, 32, 4))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, 2, 32, 4))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0  # Counter for main buffer

        # Priority buffer configuration
        self.priority_size = int(args.batch_size * k)  # Size of priority buffer
        # Priority buffer arrays for storing high-priority transitions
        self.priority_s = np.zeros((self.priority_size, 2, 32, 4))
        self.priority_a = np.zeros((self.priority_size, args.action_dim))
        self.priority_a_logprob = np.zeros((self.priority_size, args.action_dim))
        self.priority_r = np.zeros((self.priority_size, 1))
        self.priority_s_ = np.zeros((self.priority_size, 2, 32, 4))
        self.priority_dw = np.zeros((self.priority_size, 1))
        self.priority_done = np.zeros((self.priority_size, 1))
        self.priority_count = 0  # Counter for priority buffer

        self.priority_ratio = priority_ratio  # Ratio of priority samples in batch

    def store(self, s, a, a_logprob, r, s_, dw, done):
        """Store a transition in the main replay buffer"""
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1  # Increment buffer counter

    def update_priority_buffer(self):
        """Update the priority buffer with highest-priority transitions"""
        if self.count == self.batch_size:  # Only update when main buffer is full
            # Calculate priorities (simplified: use rewards as priority)
            priorities = self.r.flatten()
            # Get indices of highest-priority transitions
            top_indices = np.argsort(priorities)[-self.priority_size:]

            # Update priority buffer with highest-priority transitions
            self.priority_s = self.s[top_indices]
            self.priority_a = self.a[top_indices]
            self.priority_a_logprob = self.a_logprob[top_indices]
            self.priority_r = self.r[top_indices]
            self.priority_s_ = self.s_[top_indices]
            self.priority_dw = self.dw[top_indices]
            self.priority_done = self.done[top_indices]
            self.priority_count = self.priority_size  # Update priority buffer count

    def numpy_to_tensor(self):
        """Convert buffer data to tensors with mixed sampling (priority + normal)"""
        # Calculate batch sizes for priority and normal sampling
        priority_batch_size = int(self.batch_size * self.priority_ratio)
        normal_batch_size = self.batch_size - priority_batch_size

        # Randomly sample from priority buffer
        priority_indices = np.random.choice(self.priority_count, priority_batch_size)
        # Randomly sample from main buffer
        normal_indices = np.random.choice(self.count, normal_batch_size)

        # Extract priority samples
        s_priority = self.priority_s[priority_indices]
        a_priority = self.priority_a[priority_indices]
        a_logprob_priority = self.priority_a_logprob[priority_indices]
        r_priority = self.priority_r[priority_indices]
        s__priority = self.priority_s_[priority_indices]
        dw_priority = self.priority_dw[priority_indices]
        done_priority = self.priority_done[priority_indices]

        # Extract normal samples
        s_normal = self.s[normal_indices]
        a_normal = self.a[normal_indices]
        a_logprob_normal = self.a_logprob[normal_indices]
        r_normal = self.r[normal_indices]
        s__normal = self.s_[normal_indices]
        dw_normal = self.dw[normal_indices]
        done_normal = self.done[normal_indices]

        # Concatenate priority and normal samples
        s = np.concatenate([s_priority, s_normal], axis=0)
        a = np.concatenate([a_priority, a_normal], axis=0)
        a_logprob = np.concatenate([a_logprob_priority, a_logprob_normal], axis=0)
        r = np.concatenate([r_priority, r_normal], axis=0)
        s_ = np.concatenate([s__priority, s__normal], axis=0)
        dw = np.concatenate([dw_priority, dw_normal], axis=0)
        done = np.concatenate([done_priority, done_normal], axis=0)

        # Convert numpy arrays to torch tensors
        s = torch.tensor(s, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.float)
        a_logprob = torch.tensor(a_logprob, dtype=torch.float)
        r = torch.tensor(r, dtype=torch.float)
        s_ = torch.tensor(s_, dtype=torch.float)
        dw = torch.tensor(dw, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done  # Return tensor-formatted transitions

    def reset_buffer(self):
        """Reset both main and priority buffers to initial state"""
        # Reset main buffer
        self.s = np.zeros((self.batch_size, 2, 32, 4))
        self.a = np.zeros((self.batch_size, self.action_dim))
        self.a_logprob = np.zeros((self.batch_size, self.action_dim))
        self.r = np.zeros((self.batch_size, 1))
        self.s_ = np.zeros((self.batch_size, 2, 32, 4))
        self.dw = np.zeros((self.batch_size, 1))
        self.done = np.zeros((self.batch_size, 1))
        self.count = 0

        # Reset priority buffer
        self.priority_s = np.zeros((self.priority_size, 2, 32, 4))
        self.priority_a = np.zeros((self.priority_size, self.action_dim))
        self.priority_a_logprob = np.zeros((self.priority_size, self.action_dim))
        self.priority_r = np.zeros((self.priority_size, 1))
        self.priority_s_ = np.zeros((self.priority_size, 2, 32, 4))
        self.priority_dw = np.zeros((self.priority_size, 1))
        self.priority_done = np.zeros((self.priority_size, 1))
        self.priority_count = 0