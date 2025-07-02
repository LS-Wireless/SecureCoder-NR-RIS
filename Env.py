import copy
import torch
import numpy as np
from norm import MeanStdNormalizer, MinMaxNormalizer, LogNormalizer, DynamicNormalizer


class NR_RIS_Env(object):
    ''' '''
    def __init__(self):
        """Initialize communication environment with NR-RIS parameters"""
        # System fixed parameters
        self.N = 64  # Number of RIS elements
        self.M = 32  # Number of base station antennas
        self.K = 4  # Number of users
        self.B = 1  # Bandwidth (MHz)
        self.pho = 0.01  # Transmission power
        self.kappa = 6  # Rice factor for user-RIS channel
        self.kappa_rb = 12  # Rice factor for RIS-base station channel
        self.alpha_kb = 3.5  # Path loss exponent for user-base station direct link
        self.alpha_kr = 2.5  # Path loss exponent for user-RIS link
        self.alpha_rb = 2  # Path loss exponent for RIS-base station link
        self.kappa_kb = 3  # Rice factor for user-base station channel
        self.noise = 1e-12  # Noise power
        self.posA = np.array([35.0, 150.0, 20.0])  # Base station position
        self.posB = np.array([30.0, 145.0, 15.0])  # RIS position
        # User positions
        self.pos1 = np.array([37.0, 122.0, 2.0])
        self.pos2 = np.array([39.0, 124.0, 2.0])
        self.pos3 = np.array([41.0, 126.0, 2.0])
        self.pos4 = np.array([43.0, 128.0, 2.0])
        self.posU = [self.pos1, self.pos2, self.pos3, self.pos4]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def reset(self):
        """Reset environment and return initial state"""
        state = self.get_state()
        return state

    def get_state(self):
        """Generate and normalize channel state as environment observation"""
        self.Hur, self.Hub, self.Hrb = self.generate_channel()
        self.uplink = self.uplink_channel_compute(self.Hur, self.Hub, self.Hrb, self.nr_ris)

        # Extract real and imaginary parts of channel
        real_part = np.abs(self.uplink)
        imag_part = np.angle(self.uplink)
        # real_part = np.real(self.uplink)
        # imag_part = np.imag(self.uplink)

        # Initialize and apply normalizers
        real_normalizer = MeanStdNormalizer()
        imag_normalizer = MeanStdNormalizer()
        real_normalizer.fit(real_part)
        imag_normalizer.fit(imag_part)
        real_part_norm = real_normalizer.transform(real_part)
        imag_part_norm = imag_normalizer.transform(imag_part)

        # Combine normalized real and imaginary parts as state
        combined = np.stack((real_part_norm, imag_part_norm))
        return combined

    def step(self, action, step):
        """Execute action in environment and return next state, rewards"""
        step += 1
        H_down_real = self.downlink_channel_compute(self.Hur, self.Hub, self.Hrb, self.nr_ris)
        reward = self.compute_reward(H_down_real, self.action_w_normal(action))


        # Get next state
        state_ = self.get_state()

        # Normalize reward
        reward_normalizer = DynamicNormalizer(window_size=4000)
        reward = reward_normalizer.normalize(reward)

        return state_, reward


    def get_NR_array(self):
        """Select NR-RIS configuration based on channel data"""
        rate = 0
        best_nr_ris = None
        min_rate = float('inf')

        # Evaluate different RIS configurations
        for j in range(1):
            pair_order = np.random.permutation(self.N)
            nr_ris = np.zeros((self.N, self.N), dtype=complex)
            for i in range(self.N // 2):
                theta1 = np.exp(2j * np.pi * np.random.randn())
                nr_ris[pair_order[2 * i], pair_order[2 * i - 1]] = theta1
                nr_ris[pair_order[2 * i - 1], pair_order[2 * i]] = -theta1
            nr_ris = (nr_ris - np.eye(self.N, dtype=np.complex128))

            # Calculate performance for this configuration
            for i in range(1):
                Hur, Hub, Hrb = self.generate_channel()
                reward_direct_mrt = self.compute_reward_attack(Hub.T, Hub.T)
                reward_direct_zf = self.compute_reward_attack_zf(Hub.T, Hub.T)
                uplink = self.uplink_channel_compute(Hur, Hub, Hrb, nr_ris)
                H_down = uplink.T
                H_down_real = self.downlink_channel_compute(Hur, Hub, Hrb, nr_ris)
                rate += ((self.compute_reward_attack(H_down_real, H_down) / reward_direct_mrt) +
                         (self.compute_reward_attack_zf(H_down_real, H_down) / reward_direct_zf))

            # Update best configuration
            if rate < min_rate:
                best_nr_ris = nr_ris
                min_rate = rate
            rate = 0

        self.nr_ris = best_nr_ris

    def distance(self, x, y):
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(x - y)

    def angle(self, x, y):
        """Calculate horizontal angle between two points"""
        numerator = np.sqrt((y[0] - x[0]) ** 2)
        denominator = np.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2)
        return np.arccos(numerator / denominator)

    def user_ris_channel(self, AoA_kr, dkr):
        """Generate user-RIS channel with Rice fading"""
        indices = np.arange(self.N).reshape(-1, 1)
        hkrN = np.random.normal(0, np.sqrt(0.5), (self.N, 1)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.N, 1))
        hkrL = np.exp(-1j * np.pi * indices * np.cos(AoA_kr))
        h_kr = np.sqrt(self.pho * (dkr ** (-self.alpha_kr))) * (
                (np.sqrt(self.kappa / (1 + self.kappa)) * hkrL) + (np.sqrt(1 / (1 + self.kappa)) * hkrN))
        return np.real(h_kr), np.imag(h_kr)

    def user_base_channel(self, AoA_kb, dkb):
        """Generate user-base station channel with Rice fading"""
        indices = np.arange(self.M).reshape(-1, 1)
        hubN = np.random.normal(0, np.sqrt(0.5), (self.M, 1)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.M, 1))
        hubL = np.exp(-1j * np.pi * indices * np.cos(AoA_kb))
        h_ub = np.sqrt(self.pho * (dkb ** (-self.alpha_kb))) * (
                (np.sqrt(self.kappa_kb / (1 + self.kappa_kb)) * hubL) + (np.sqrt(1 / (1 + self.kappa_kb)) * hubN))
        return np.real(h_ub), np.imag(h_ub)

    def ris_base_channel(self, drb, AoA_rb, AoD_rb):
        """Generate RIS-base station channel with Rice fading"""
        indices = np.arange(self.N).reshape(-1, 1)
        indices2 = np.arange(self.M).reshape(-1, 1)
        H_rbN = np.random.normal(0, np.sqrt(0.5), (self.M, self.N)) + 1j * np.random.normal(0, np.sqrt(0.5),
                                                                                            (self.M, self.N))
        H_rbL = (np.exp(-1j * np.pi * indices2 * np.cos(AoA_rb))) @ (
            (np.exp(-1j * np.pi * indices * np.cos(AoD_rb))).conjugate().T)
        H_rb = np.sqrt(self.pho * (drb ** (-self.alpha_rb))) * (
                (np.sqrt(self.kappa_rb / (1 + self.kappa_rb)) * H_rbL) + (np.sqrt(1 / (1 + self.kappa_rb)) * H_rbN))
        return H_rb

    def generate_channel(self):
        """Generate all communication channels: user-RIS, user-base, RIS-base"""
        Hur_real = np.empty([self.N, self.K])
        Hur_imag = np.empty([self.N, self.K])
        Hub_real = np.empty([self.M, self.K])
        Hub_imag = np.empty([self.M, self.K])

        # RIS-base station channel
        drb = self.distance(self.posA, self.posB)
        AoA_rb = self.angle(self.posB, self.posA)
        AoD_rb = AoA_rb
        Hrb = self.ris_base_channel(drb, AoA_rb, AoD_rb)

        # User channels
        for i in range(self.K):
            pos = self.posU[i]
            dkr = self.distance(self.posB, pos)
            dkb = self.distance(self.posA, pos)
            AoA_kr = self.angle(pos, self.posB)
            AoA_kb = self.angle(pos, self.posA)
            hkr_real, hkr_imag = self.user_ris_channel(AoA_kr, dkr)
            hkb_real, hkb_imag = self.user_base_channel(AoA_kb, dkb)

            Hur_real[:, i] = hkr_real.squeeze()
            Hur_imag[:, i] = hkr_imag.squeeze()
            Hub_real[:, i] = hkb_real.squeeze()
            Hub_imag[:, i] = hkb_imag.squeeze()

        return Hur_real + 1j * Hur_imag, Hub_real + 1j * Hub_imag, Hrb

    def uplink_channel_compute(self, Hur, Hub, Hrb, nr_ris):
        """Compute uplink channel: RIS-assisted + direct"""
        return (Hrb @ nr_ris @ Hur) + Hub

    def downlink_channel_compute(self, Hur, Hub, Hrb, nr_ris):
        """Compute downlink channel: RIS-assisted + direct"""
        return (Hur.T @ nr_ris @ Hrb.T) + Hub.T

    def compute_reward(self, H_real_down, action):
        """Compute reward based on  sum rate of users by SecureCode"""
        W = action
        rate = np.zeros(self.K)
        for k in range(self.K):
            sig = (np.abs(np.dot(H_real_down[k, :], W[:, k])) ** 2)
            W_removed = np.delete(W, k, axis=1)
            interference = np.sum(np.abs(np.dot(H_real_down[k, :], W_removed)) ** 2)
            sinr = sig / (interference + self.noise)
            rate[k] = 10 * self.B * np.log2(1 + sinr) # Times 10 is for scaling reward value. During testing, we will remove this 10.
            rate[k] = np.log2(rate[k] + 1e-5)  # Avoid log(0)
        return np.sum(rate)

    def compute_reward_attack(self, H_real_down, H_down):
        """Compute reward for MRT precoding attack"""
        W = self.MRT_precoding(H_down)
        rate = np.zeros(self.K)
        for k in range(self.K):
            sig = (np.abs(np.dot(H_real_down[k, :], W[:, k])) ** 2)
            W_removed = np.delete(W, k, axis=1)
            interference = np.sum(np.abs(np.dot(H_real_down[k, :], W_removed)) ** 2)
            sinr = sig / (interference + self.noise)
            rate[k] = self.B * np.log2(1 + sinr)
        return np.sum(rate)

    def compute_reward_attack_zf(self, H_real_down, H_down):
        """Compute reward for ZF precoding attack"""
        W = self.ZF_precoding(H_down)
        rate = np.zeros(self.K)
        for k in range(self.K):
            sig = np.abs(np.dot(H_real_down[k, :], W[:, k])) ** 2
            W_removed = np.delete(W, k, axis=1)
            interference = np.sum(np.abs(np.dot(H_real_down[k, :], W_removed)) ** 2)
            sinr = sig / (interference + self.noise)
            rate[k] = self.B * np.log2(1 + sinr)
        return np.sum(rate)

    def ZF_precoding(self, H):
        """Zero-Forcing precoding implementation"""
        H_hermitian = np.conj(H).T
        W_zf = np.linalg.pinv(H_hermitian @ H) @ H_hermitian
        return W_zf / np.linalg.norm(W_zf, 'fro')  # Normalize

    def MRT_precoding(self, H):
        """Maximum Ratio Transmission precoding implementation"""
        H_hermitian = np.conj(H).T
        return H_hermitian / np.linalg.norm(H_hermitian, 'fro')  # Normalize

    def action_w_normal(self, action):
        """Convert and normalize agent action to complex precoding matrix,satisfy power constraint """
        w = copy.deepcopy(action)
        w[self.M * self.K:] = w[self.M * self.K:] * 2 * np.pi  # Map to [0, 2π]
        action_w = w.reshape((2, self.M, self.K))
        # Convert amplitude and phase to complex numbers
        action_w = np.vectorize(lambda amp, phase: amp * (np.cos(phase) + 1j * np.sin(phase)))(
            action_w[0], action_w[1])
        return action_w / np.linalg.norm(action_w, 'fro')  # Normalize

    def action_w_random(self):
        """Generate random precoding matrix for baseline comparison"""
        w = np.array(np.random.randn(2 * self.M * self.K).tolist())
        w[self.M * self.K:] = w[self.M * self.K:] * 2 * np.pi  # Map to [0, 2π]
        action_w = w.reshape((2, self.M, self.K))
        # Convert amplitude and phase to complex numbers
        action_w = np.vectorize(lambda amp, phase: amp * (np.cos(phase) + 1j * np.sin(phase)))(
            action_w[0], action_w[1])
        return action_w / np.linalg.norm(action_w, 'fro')  # Normalize
