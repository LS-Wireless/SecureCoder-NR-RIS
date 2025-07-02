import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal



# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.conv_1 = nn.Conv2d(2, 4, kernel_size=2, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(4, 8, kernel_size=2, stride=1, padding=1)
        self.fc1 = nn.Linear(1632, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)

        self.alpha_layer = nn.Linear(512, args.action_dim)
        self.beta_layer = nn.Linear(512, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  #  use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        H_matrix = s
        H_matrix = F.relu(self.conv_1(H_matrix))
        H_matrix = F.relu(self.conv_2(H_matrix))
        s = H_matrix.view(H_matrix.size(0), -1)
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        s = self.activate_func(self.fc3(s))

        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        # print('Value of softplus: ', F.softplus(self.alpha_layer(s)))
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta


    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean

class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()

        self.conv_real = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv_imag = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv_matrix = nn.Conv2d(4, 16, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(1024, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)

        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):

        H_matrix = s
        H_matrix = F.relu(self.conv_real(H_matrix))
        s = torch.flatten(H_matrix, 1)
        s = self.activate_func(self.fc1(s))

        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        # print('Value of softplus: ', F.softplus(self.alpha_layer(s)))
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        self.conv_1= nn.Conv2d(2, 4, kernel_size=2, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(4, 8, kernel_size=2, stride=1, padding=1)
        self.fc1 = nn.Linear(1632, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)


        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        H_matrix =s
        H_matrix = F.relu(self.conv_1(H_matrix))
        H_matrix = F.relu(self.conv_2(H_matrix))
        s = H_matrix.view(H_matrix.size(0), -1)
        s = self.activate_func(self.fc1(s))

        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_continuous_cnn():
    def __init__(self, args):
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.device = torch.device("cuda")
        self.max_train_episode  = args.max_train_episode
        self.total_step = 0

        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args).to(self.device)
        else:
            self.actor = Actor_Gaussian(args).to(self.device)
        self.critic = Critic(args).to(self.device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        # s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        # print(s)
        s = torch.tensor(s, dtype=torch.float).to(torch.device("cuda")).unsqueeze(0)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).cpu().detach().numpy().flatten()
        else:
            a = self.actor(s).cpu().detach().numpy().flatten()
        return a

    def choose_action(self, s):
        # s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        s = torch.tensor(s, dtype=torch.float).to(torch.device("cuda")).unsqueeze(0)
        # print('-------------')
        # print(s)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def update(self, replay_buffer,step):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor() # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        self.total_step +=0
        # Move tensors to GPU
        s = s.to(self.device)
        a = a.to(self.device)
        a_logprob = a_logprob.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        dw = dw.to(self.device)
        done = done.to(self.device)
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(self.total_step)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - (0.95*total_steps) / (self.max_train_episode * self.max_train_steps))
        lr_c_now = self.lr_c * (1 - (0.95*total_steps) / (self.max_train_episode* self.max_train_steps))
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def save_model(self, env_name):
        torch.save(self.actor.state_dict(), "./actor_{}.pth".format(env_name))
        torch.save(self.critic.state_dict(), "./critic_{}.pth".format(env_name))
        print("save model success")

    def load_model(self, env_name):
        self.actor.load_state_dict(torch.load("./actor_{}.pth".format(env_name)))



