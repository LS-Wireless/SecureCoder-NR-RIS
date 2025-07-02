import copy
import torch
import numpy as np
from norm import MeanStdNormalizer, MinMaxNormalizer, LogNormalizer, DynamicNormalizer


class NR_RIS_Env(object):
    def __init__(self, N, M):
        """Initialize the RIS-assisted communication environment"""
        # System parameters
        self.N = N  # Number of RIS elements
        self.M = M  # Number of base station antennas
        self.K = 4  # Number of users
        self.B = 1  # Bandwidth (MHz)
        self.pho = 0.01  # Transmission power
        self.kappa = 6  # Rice factor for user-RIS channel
        self.kappa_rb = 12  # Rice factor for RIS-base station channel
        self.alpha_kb = 3.5  # Path loss exponent for direct user-base station link
        self.alpha_kr = 2.5  # Path loss exponent for user-RIS link
        self.alpha_rb = 2  # Path loss exponent for RIS-base station link
        self.noise = 1e-12  # Noise power
        self.posA = np.array([35.0, 150.0, 20.0])  # Base station position
        self.kappa_kb = 3  # Rice factor for user-base station channel
        self.posB = np.array([30.0, 145.0, 15.0])  # RIS position
        self.pose = np.array([36.0, 125.0, 2.0])  # Eavesdropper position
        self.kappa_eb = 4  # Rice factor for eavesdropper-base station channel
        self.kappa_er = 8  # Rice factor for eavesdropper-RIS channel
        self.alpha_eb = 3.2  # Path loss exponent for eavesdropper-base station link
        self.alpha_er = 2.5  # Path loss exponent for eavesdropper-RIS link

        # User positions
        self.pos1 = np.array([37.0, 122.0, 2.0])
        self.pos2 = np.array([39.0, 124.0, 2.0])
        self.pos3 = np.array([41.0, 126.0, 2.0])
        self.pos4 = np.array([43.0, 128.0, 2.0])
        self.posU = [self.pos1, self.pos2, self.pos3, self.pos4]

        # Device configuration
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Outage counters for different strategies
        self.out_secure = 0
        self.out_random_bf = 0
        self.out_NOcrack_mrt = 0
        self.out_NOcrack_zf = 0
        self.out_crack_mrt = 0
        self.out_crack_zf = 0
        self.out_random_mrt = 0
        self.out_random_zf = 0

    def reset(self):
        """Reset the environment and return the initial state"""
        state = self.get_state()
        return state

    def get_state(self):
        """Generate channel state and normalize it as the environment observation"""
        self.Hur, self.Hub, self.Hrb = self.generate_channel()
        self.uplink = self.uplink_channel_compute(self.Hur, self.Hub, self.Hrb, self.nr_ris)

        # Extract real and imaginary parts of the uplink channel
        real_part = np.abs(self.uplink)
        imag_part = np.angle(self.uplink)

        # real_part = np.real(self.uplink)
        # imag_part = np.imag(self.uplink)


        # Normalize the channel data
        real_normalizer = MeanStdNormalizer()
        imag_normalizer = MeanStdNormalizer()
        real_normalizer.fit(real_part)
        imag_normalizer.fit(imag_part)
        real_part_norm = real_normalizer.transform(real_part)
        imag_part_norm = imag_normalizer.transform(imag_part)

        # Combine normalized real and imaginary parts
        combined = np.stack((real_part_norm, imag_part_norm))
        return combined

    def step_test(self, action, step):
        """Test step for evaluating different policies against eavesdropping"""
        step += 1
        self.H_er, self.H_eb = self.generate_eve_channel()
        H_down_real = self.downlink_channel_compute(self.Hur, self.Hub, self.Hrb, self.nr_ris)
        H_down_real_eve = self.downlink_channel_compute(self.H_er, self.H_eb, self.Hrb, self.nr_ris)

        # Calculate rewards for legitimate users
        reward, rate_secure = self.compute_reward(H_down_real, self.action_w_normal(action))
        # Calculate rewards for the eavesdropper
        reward_eve_secure, rate_eve_secure = self.compute_reward_eve(H_down_real_eve, self.action_w_normal(action))
        reward_eve_random_bf, rate_eve_random_bf = self.compute_reward_eve(H_down_real_eve, self.action_w_random())
        reward_eve_crack_mrt, rate_eve_crack_mrt = self.compute_reward_eve(H_down_real_eve,
                                                                           self.MRT_precoding(self.uplink.T))
        reward_eve_crack_zf, rate_eve_crack_zf = self.compute_reward_eve(H_down_real_eve,
                                                                         self.ZF_precoding(self.uplink.T))


        # Calculate rewards for baseline strategies
        reward_random_bf, rate_random_bf = self.compute_reward(H_down_real, self.action_w_random())
        reward_mrt, rate_mrt = self.compute_reward_attack(H_down_real, self.uplink.T)
        reward_zf, rate_zf = self.compute_reward_attack_zf(H_down_real, self.uplink.T)
        reward_direct_mrt, rate_direct_mrt = self.compute_reward_attack(self.Hub.T, self.Hub.T)
        reward_direct_zf, rate_direct_zf = self.compute_reward_attack_zf(self.Hub.T, self.Hub.T)

        # Calculate secure rates by subtracting eavesdropper rates
        reward_secure = sum(rate_secure[i] - rate_eve_secure[i] for i in range(self.K))
        reward_random_bf_secure = sum(max(rate_random_bf[i] - rate_eve_random_bf[i], 0) for i in range(self.K))
        reward_crack_mrt_secure = sum(max(rate_mrt[i] - rate_eve_crack_mrt[i], 0) for i in range(self.K))
        reward_crack_zf_secure = sum(max(rate_zf[i] - rate_eve_crack_zf[i], 0) for i in range(self.K))

        # Generate a random RIS configuration for testing
        pair_order = np.random.permutation(self.N)
        nr_ris = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N // 2):
            theta1 = np.exp(2j * np.pi * np.random.randn())
            nr_ris[pair_order[2 * i], pair_order[2 * i - 1]] = theta1
            nr_ris[pair_order[2 * i - 1], pair_order[2 * i]] = -theta1
        nr_ris = (nr_ris - np.eye(self.N, dtype=np.complex128))

        uplink = self.uplink_channel_compute(self.Hur, self.Hub, self.Hrb, nr_ris)
        H_down_real_random_ris = self.downlink_channel_compute(self.Hur, self.Hub, self.Hrb, nr_ris)
        reward_random_mrt, rate_random_mrt = self.compute_reward_attack(H_down_real_random_ris, uplink.T)
        reward_random_zf, rate_random_zf = self.compute_reward_attack_zf(H_down_real_random_ris, uplink.T)

        # Update outage counters
        for i in range(len(rate_secure)):
            if rate_secure[i] < rate_eve_secure[i]:
                self.out_secure += 1
            if rate_random_bf[i] < rate_eve_random_bf[i]:
                self.out_random_bf += 1
            if rate_mrt[i] < rate_eve_crack_mrt[i]:
                self.out_crack_mrt += 1
            if rate_zf[i] < rate_eve_crack_zf[i]:
                self.out_crack_zf += 1

        return (None, np.max(reward - reward_eve_secure, 0), reward_secure, reward_random_bf_secure,
                reward_crack_mrt_secure, reward_crack_zf_secure, reward, reward_random_bf,
                reward_random_mrt, reward_random_zf, reward_mrt, reward_direct_mrt, reward_zf, reward_direct_zf)

    def step_test_disco(self, action, step):
        """Extended test step for evaluating against random RIS configurations"""
        step += 1
        # Initialize reward accumulators
        reward_disco = 0
        reward_eve_secure_disco = 0
        reward_eve_random_bf_disco = 0
        reward_eve_crack_mrt_disco = 0
        reward_eve_crack_zf_disco = 0
        reward_random_bf_disco = 0
        reward_mrt_disco = 0
        reward_zf_disco = 0
        reward_random_mrt_disco = 0
        reward_random_zf_disco = 0
        reward_direct_mrt_disco = 0
        reward_direct_zf_disco = 0
        reward_secure_disco = 0
        reward_random_bf_secure_disco = 0
        reward_crack_mrt_secure_disco = 0
        reward_crack_zf_secure_disco = 0

        for i in range(1):
            # Generate a random diagonal RIS matrix
            matrix = np.random.randn(self.N, 1)
            complex_matrix = np.exp(2j * np.pi * matrix)
            diagonal_matrix = np.diag(np.squeeze(complex_matrix))
            identity_matrix = np.eye(self.N, dtype=np.complex128)
            nr_ris = diagonal_matrix - identity_matrix

            # Generate eavesdropper channels
            self.H_er, self.H_eb = self.generate_eve_channel()
            H_down_real = self.downlink_channel_compute(self.Hur, self.Hub, self.Hrb, nr_ris)
            H_down_real_eve = self.downlink_channel_compute(self.H_er, self.H_eb, self.Hrb, nr_ris)

            # Calculate rewards for different strategies
            reward, rate_secure = self.compute_reward(H_down_real, self.action_w_normal(action))
            reward_eve_secure, rate_eve_secure = self.compute_reward_eve(H_down_real_eve, self.action_w_normal(action))
            reward_eve_random_bf, rate_eve_random_bf = self.compute_reward_eve(H_down_real_eve, self.action_w_random())
            reward_eve_crack_mrt, rate_eve_crack_mrt = self.compute_reward_eve(H_down_real_eve,
                                                                               self.MRT_precoding(self.uplink.T))
            reward_eve_crack_zf, rate_eve_crack_zf = self.compute_reward_eve(H_down_real_eve,
                                                                             self.ZF_precoding(self.uplink.T))
            reward_random_bf, rate_random_bf = self.compute_reward(H_down_real, self.action_w_random())
            reward_mrt, rate_mrt = self.compute_reward_attack(H_down_real, self.uplink.T)
            reward_zf, rate_zf = self.compute_reward_attack_zf(H_down_real, self.uplink.T)
            reward_direct_mrt, rate_direct_mrt = self.compute_reward_attack(self.Hub.T, self.Hub.T)
            reward_direct_zf, rate_direct_zf = self.compute_reward_attack_zf(self.Hub.T, self.Hub.T)

            # Accumulate rewards
            reward_disco += reward
            reward_eve_secure_disco += reward_eve_secure
            reward_eve_random_bf_disco += reward_eve_random_bf
            reward_eve_crack_mrt_disco += reward_eve_crack_mrt
            reward_eve_crack_zf_disco += reward_eve_crack_zf
            reward_random_bf_disco += reward_random_bf
            reward_mrt_disco += reward_mrt
            reward_zf_disco += reward_zf
            reward_direct_mrt_disco += reward_direct_mrt
            reward_direct_zf_disco += reward_direct_zf

            # Calculate secure rates
            for i in range(self.K):
                reward_secure_disco += rate_secure[i] - rate_eve_secure[i]
                reward_random_bf_secure_disco += max((rate_random_bf[i] - rate_eve_random_bf[i]), 0)
                reward_crack_mrt_secure_disco += max((rate_mrt[i] - rate_eve_crack_mrt[i]), 0)
                reward_crack_zf_secure_disco += max((rate_zf[i] - rate_eve_crack_zf[i]), 0)

            # Test another random RIS configuration
            matrix = np.random.randn(self.N, self.N) + 1j * np.random.randn(self.N, self.N)
            nr_ris_random = matrix - identity_matrix
            uplink_random = self.uplink_channel_compute(self.Hur, self.Hub, self.Hrb, nr_ris_random)
            H_down_real_random_ris = self.downlink_channel_compute(self.Hur, self.Hub, self.Hrb, nr_ris_random)
            reward_random_mrt, rate_random_mrt = self.compute_reward_attack(H_down_real_random_ris, uplink_random.T)
            reward_random_zf, rate_random_zf = self.compute_reward_attack_zf(H_down_real_random_ris, uplink_random.T)
            reward_random_mrt_disco += reward_random_mrt
            reward_random_zf_disco += reward_random_zf

        return (None, reward_disco - reward_eve_crack_mrt_disco, reward_secure_disco,
                reward_random_bf_secure_disco, reward_crack_mrt_secure_disco, reward_crack_zf_secure_disco,
                reward_disco, reward_random_bf_disco, reward_random_mrt_disco, reward_random_zf_disco,
                reward_mrt_disco, reward_direct_mrt_disco, reward_zf_disco, reward_direct_zf_disco)

    def get_NR_array(self, malicious):
        """Select the NR-RIS configuration based on channel data"""
        t =50 if malicious else 1  # t=N represents select one malicious NR-RIS from N randomly generated NR-RIS values. Since it influences the test speed, please select a reasonable value
        rate = 0
        min_rate = float('inf')
        list = []
        for j in range(t):
            # Generate a random RIS configuration
            pair_order = np.random.permutation(self.N)
            nr_ris = np.zeros((self.N, self.N), dtype=complex)
            for i in range(self.N // 2):
                theta1 = np.exp(2j * np.pi * np.random.randn())
                nr_ris[pair_order[2 * i], pair_order[2 * i - 1]] = theta1
                nr_ris[pair_order[2 * i - 1], pair_order[2 * i]] = -theta1
            nr_ris = (nr_ris - np.eye(self.N, dtype=np.complex128))

            # Evaluate the configuration using channel data
            for i in range(t):
                Hur, Hub, Hrb= self.generate_channel()
                reward_direct_mrt, rate_direct_mrt = self.compute_reward_attack(Hub.T, Hub.T)
                reward_direct_zf, rate_direct_zf = self.compute_reward_attack_zf(Hub.T, Hub.T)
                uplink = self.uplink_channel_compute(Hur, Hub, Hrb, nr_ris)
                H_down = uplink.T
                H_down_real = self.downlink_channel_compute(Hur, Hub, Hrb, nr_ris)
                reward_mrt, rate_mrt = self.compute_reward_attack(H_down_real, H_down)
                reward_zf, rate_zf = self.compute_reward_attack_zf(H_down_real, H_down)
                rate += ((reward_mrt / reward_direct_mrt) + (reward_zf / reward_direct_zf)) # Malicious NR-RIS simultaneously attacks MRT and ZF

            # Keep track of the best configuration
            list.append(rate)
            if j == 0 or rate < min_rate:
                min_rate = rate
                self.nr_ris = nr_ris
            rate = 0

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
        return np.real(h_ub), np.imag(h_ub), np.real(hubL), np.imag(hubL)

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

    def generate_eve_channel(self):
        """Generate eavesdropper channels (RIS and base station)"""
        der = self.distance(self.posB, self.pose)
        AoA_kr = self.angle(self.pose, self.posB)
        her_real, her_imag = self.eve_ris_channel(AoA_kr, der)
        Her = her_real + 1j * her_imag

        deb = self.distance(self.posA, self.pose)
        AoA_eb = self.angle(self.pose, self.posA)
        heb_real, heb_imag = self.eve_base_channel(AoA_eb, deb)
        Heb = heb_real + 1j * heb_imag
        return Her, Heb

    def eve_ris_channel(self, AoA_kr, dkr):
        """Generate eavesdropper-RIS channel with Rice fading"""
        indices = np.arange(self.N).reshape(-1, 1)
        hkrN = np.random.normal(0, np.sqrt(0.5), (self.N, 1)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.N, 1))
        hkrL = np.exp(-1j * np.pi * indices * np.cos(AoA_kr))
        h_kr = np.sqrt(self.pho * (dkr ** (-self.alpha_er))) * (
                (np.sqrt(self.kappa_er / (1 + self.kappa_er)) * hkrL) + (np.sqrt(1 / (1 + self.kappa_er)) * hkrN))
        return np.real(h_kr), np.imag(h_kr)

    def eve_base_channel(self, AoA_kb, dkb):
        """Generate eavesdropper-base station channel with Rice fading"""
        indices = np.arange(self.M).reshape(-1, 1)
        hubN = np.random.normal(0, np.sqrt(0.5), (self.M, 1)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.M, 1))
        hubL = np.exp(-1j * np.pi * indices * np.cos(AoA_kb))
        h_ub = np.sqrt(self.pho * (dkb ** (-self.alpha_eb))) * (
                (np.sqrt(self.kappa_eb / (1 + self.kappa_eb)) * hubL) + (np.sqrt(1 / (1 + self.kappa_eb)) * hubN))
        return np.real(h_ub), np.imag(h_ub)

    def generate_channel(self):
        """Generate all communication channels (user-RIS, user-base, RIS-base)"""
        Hur_real = np.empty([self.N, self.K])
        Hur_imag = np.empty([self.N, self.K])
        Hub_real = np.empty([self.M, self.K])
        Hub_imag = np.empty([self.M, self.K])
        HubL_real = np.empty([self.M, self.K])
        HubL_imag = np.empty([self.M, self.K])

        # Generate RIS-base station channel
        drb = self.distance(self.posA, self.posB)
        AoA_rb = self.angle(self.posB, self.posA)
        AoD_rb = AoA_rb
        Hrb = self.ris_base_channel(drb, AoA_rb, AoD_rb)

        # Generate user channels
        for i in range(self.K):
            pos = self.posU[i]
            dkr = self.distance(self.posB, pos)
            dkb = self.distance(self.posA, pos)
            AoA_kr = self.angle(pos, self.posB)
            AoA_kb = self.angle(pos, self.posA)
            hkr_real, hkr_imag = self.user_ris_channel(AoA_kr, dkr)
            hkb_real, hkb_imag, hkbL_real, hkbL_imag = self.user_base_channel(AoA_kb, dkb)

            # Populate channel matrices
            Hur_real[:, i] = hkr_real.squeeze()
            Hur_imag[:, i] = hkr_imag.squeeze()
            Hub_real[:, i] = hkb_real.squeeze()
            Hub_imag[:, i] = hkb_imag.squeeze()
            HubL_real[:, i] = hkbL_real.squeeze()
            HubL_imag[:, i] = hkbL_imag.squeeze()

        return Hur_real + 1j * Hur_imag, Hub_real + 1j * Hub_imag, Hrb
    def uplink_channel_compute(self, Hur, Hub, Hrb, nr_ris):
        """Compute the uplink channel (RIS-assisted + direct)"""
        return (Hrb @ nr_ris @ Hur) + Hub

    def downlink_channel_compute(self, Hur, Hub, Hrb, nr_ris):
        """Compute the downlink channel (RIS-assisted + direct)"""
        return (Hur.T @ nr_ris @ Hrb.T) + Hub.T

    def compute_reward(self, H_real_down, action):
        """Compute reward based on the sum rate of legitimate users"""
        W = action
        rate = np.zeros(self.K)
        for k in range(self.K):
            sig = (np.abs(np.dot(H_real_down[k, :], W[:, k])) ** 2)
            W_removed = np.delete(W, k, axis=1)
            interference = np.sum(np.abs(np.dot(H_real_down[k, :], W_removed)) ** 2)
            sinr = sig / (interference + self.noise)
            rate[k] = self.B * np.log2(1 + sinr)
        return np.sum(rate), rate

    def compute_reward_eve(self, H_real_down_eve, action):
        """Compute reward for the eavesdropper's sum rate"""
        W = action
        rate = np.zeros(self.K)
        for k in range(self.K):
            sig = (np.abs(np.dot(H_real_down_eve, W[:, k])) ** 2)
            W_removed = np.delete(W, k, axis=1)
            interference = np.sum(np.abs(np.dot(H_real_down_eve, W_removed)) ** 2)
            sinr = sig / (interference + self.noise)
            rate[k] = self.B * np.log2(1 + sinr)
        return np.sum(rate), rate

    def compute_reward_attack(self, H_real_down, H_down):
        """Compute reward using MRT precoding for attack evaluation"""
        W = self.MRT_precoding(H_down)
        rate = np.zeros(self.K)
        for k in range(self.K):
            sig = (np.abs(np.dot(H_real_down[k, :], W[:, k])) ** 2)
            W_removed = np.delete(W, k, axis=1)
            interference = np.sum(np.abs(np.dot(H_real_down[k, :], W_removed)) ** 2)
            sinr = sig / (interference + self.noise)
            rate[k] = self.B * np.log2(1 + sinr)
        return np.sum(rate), rate

    def compute_reward_attack_zf(self, H_real_down, H_down):
        """Compute reward using ZF precoding for attack evaluation"""
        W = self.ZF_precoding(H_down)
        rate = np.zeros(self.K)
        for k in range(self.K):
            sig = np.abs(np.dot(H_real_down[k, :], W[:, k])) ** 2
            W_removed = np.delete(W, k, axis=1)
            interference = np.sum(np.abs(np.dot(H_real_down[k, :], W_removed)) ** 2)
            sinr = sig / (interference + self.noise)
            rate[k] = self.B * np.log2(1 + sinr)
        return np.sum(rate), rate

    def ZF_precoding(self, H):
        """Implement Zero-Forcing precoding with normalization"""
        H_hermitian = np.conj(H).T
        W_zf = np.linalg.pinv(H_hermitian @ H) @ H_hermitian
        return W_zf / np.linalg.norm(W_zf, 'fro')

    def MRT_precoding(self, H):
        """Implement Maximum Ratio Transmission precoding with normalization"""
        H_hermitian = np.conj(H).T
        return H_hermitian / np.linalg.norm(H_hermitian, 'fro')


    def action_w_normal(self, action):
        """Convert and normalize agent action to a complex precoding matrix"""
        w = copy.deepcopy(action)
        w[self.M * self.K:] = w[self.M * self.K:] * 2 * np.pi  # Map phases to [0, 2π]
        action_w = w.reshape((2, self.M, self.K))
        # Convert amplitude and phase to complex numbers
        action_w = np.vectorize(lambda amp, phase: amp * (np.cos(phase) + 1j * np.sin(phase)))(
            action_w[0], action_w[1])
        return action_w / np.linalg.norm(action_w, 'fro')

    def action_w_random(self):
        """Generate a random precoding matrix for baseline comparison"""
        w = np.array(np.random.randn(2 * self.M * self.K).tolist())
        w[self.M * self.K:] = w[self.M * self.K:] * 2 * np.pi  # Map phases to [0, 2π]
        action_w = w.reshape((2, self.M, self.K))
        # Convert amplitude and phase to complex numbers
        action_w = np.vectorize(lambda amp, phase: amp * (np.cos(phase) + 1j * np.sin(phase)))(
            action_w[0], action_w[1])
        return action_w / np.linalg.norm(action_w, 'fro')
