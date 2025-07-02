import torch
import numpy as np
import argparse

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# Allow duplicate KMP libraries to resolve potential runtime issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from replaybuffer_con_cnn_per import ReplayBuffer_cnn
from ppo_continuous_cnn import PPO_continuous_cnn
import Env

# Initialize basic parameters for the environment
M = 32
N = 256
K = 4

# Create environment instances for training and evaluation
env = Env.NR_RIS_Env()
env_evaluate = Env.NR_RIS_Env() # it is for generating the figure of training process

def evaluate_policy(args, env, agent, nr_ris):
    """Evaluate the policy performance """
    env.nr_ris = nr_ris
    times = 1
    evaluate_reward_all = 0

    # Run evaluation episodes
    for ep in range(times):
        state = env.reset()
        episode_reward_all = 0

        # Run each episode for max_train_steps
        for step in range(args.max_train_steps):
            # Get action from the agent
            action = agent.evaluate(state)
            # Execute action and get rewards
            state_, reward_all= env.step(action, step)
            # Accumulate reward
            episode_reward_all += reward_all
            state = state_

        # Accumulate evaluation rewards
        evaluate_reward_all += episode_reward_all

    # Return average rewards
    return evaluate_reward_all / times


def main(args, seed):
    """Main training function for the PPO agent"""


    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set state and action dimensions
    args.state_dim = 2 * M * K
    args.action_dim = 2 * M * K
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    args.max_action = float(1)

    # Initialize replay buffer and agent
    replay_buffer_continuous = ReplayBuffer_cnn(args)
    agent = PPO_continuous_cnn(args)

    # Tracking variables for evaluation
    evaluate_num = 0  # Record evaluation count
    evaluate_rewards_all = []  # Record all rewards during evaluation
    total_steps = 0

    # Set initial channel data for the environment
    env.get_NR_array()

    # Main training loop
    for ep in range(args.max_train_episode):
        # Update channel data periodically
        if ep % 100 == 0:
            env.get_NR_array()
        state = env.reset()

        reward_ep = 0

        # Run each episode for max_train_steps
        for step in range(args.max_train_steps):
            print("step: {}".format(step))
            # Get action from agent with log probability
            action, a_logprob = agent.choose_action(state)

            # Adjust action based on policy distribution
            if args.policy_dist == "Beta":
                action = action * agent.max_action
            else:
                action = action

            # Execute action in environment and get results
            state_, reward_all = env.step(action, step)
            print('reward_all', reward_all)

            # Accumulate episode rewards
            reward_ep += reward_all

            # Determine terminal state
            if step + 1 == args.max_train_steps:
                dw = True
                done = True
            else:
                dw = False
                done = False

            state = state_
            # Store transition in replay buffer
            replay_buffer_continuous.store(state, action, a_logprob, reward_all, state_, dw, done)
            # Update priority buffer
            replay_buffer_continuous.update_priority_buffer()
            total_steps += 1

            # Update agent when buffer has enough transitions
            if replay_buffer_continuous.count == args.batch_size:
                agent.update(replay_buffer_continuous, ep)
                replay_buffer_continuous.count = 0

            # Evaluate policy periodically
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward_all = evaluate_policy(args,
                                                                                                      env_evaluate,
                                                                                                      agent, env.nr_ris)
                evaluate_rewards_all.append(evaluate_reward_all)

                # Apply Savitzky-Golay filter for smoothing
                reward_filter_all = savgol_filter(evaluate_rewards_all, 53, 3, mode='nearest')

                # Plot evaluation results periodically
                if evaluate_num % 100 == 0:
                    fig, ax1 = plt.subplots(figsize=(6, 7))
                    ax1.plot(reward_filter_all, color='blue')
                    ax1.plot(evaluate_rewards_all, color='red', alpha=0.3)
                    ax1.set_xlabel('Episodes')
                    ax1.set_ylabel('Return Reward')
                    foo_fig = plt.gcf()
                    foo_fig.savefig('./train_32_64_4(entroy=0.005)(per_log2_ris2_no_fixed_rewardNorm).pdf',
                                    format='pdf', bbox_inches='tight', dpi=600, pad_inches=0.0)
                    plt.show()
        # Save model periodically
        if ep % 5000 == 0:
            agent.save_model('agent_32_64_4(entroy=0.005)(per_log2_ris2_no_fixed_rewardNorm)')
        # Save evaluation results
        np.savetxt('32_256_4(entroy=0.005)(per_log2_ris2_no_fixed_rewardNorm)', evaluate_rewards_all)


if __name__ == '__main__':
    """Parse command line arguments and start training"""
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_episode", type=int, default=int(300000),
                        help="Maximum number of training episodes")
    parser.add_argument("--max_train_steps", type=int, default=int(20),
                        help="Maximum number of training steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=200, help="Evaluate policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Model saving frequency")
    parser.add_argument("--policy_dist", type=str, default="Beta", help="Policy distribution: Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=4000, help="Batch size for training")
    parser.add_argument("--mini_batch_size", type=int, default=2000, help="Minibatch size for PPO updates")
    parser.add_argument("--hidden_width", type=int, default=512, help="Number of neurons in hidden layers")
    parser.add_argument("--lr_a", type=float, default=1e-5, help="Learning rate for actor network")
    parser.add_argument("--lr_c", type=float, default=1e-5, help="Learning rate for critic network")
    parser.add_argument("--gamma", type=float, default=0, help="Discount factor for future rewards")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE (Generalized Advantage Estimation) parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="PPO clip parameter for policy update")
    parser.add_argument("--K_epochs", type=int, default=10, help="Number of epochs for PPO updates")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1: Advantage normalization")
    parser.add_argument("--entropy_coef", type=float, default=0.005, help="Trick 5: Policy entropy coefficient")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6: Learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: Orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: Set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: Tanh activation function")
    parser.add_argument("--use_gru", type=bool, default=True, help="Whether to use GRU in the network")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Number of neurons in hidden layers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for actor")

    args = parser.parse_args()

    # Start training with specified seed
    main(args, seed=20)
