# main function that sets up environments
# perform training loop
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
import os
from utilities import transpose_list, transpose_to_tensor
from collections import deque

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    seeding()
    # number of training episodes.
    number_of_episodes = 5000
    episode_length = 1000
    batchsize = 2000
    t = 0

    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 2
    noise_reduction = 0.9999

    # how many episodes before update
    episode_per_update = 2

    log_path = os.getcwd()+"/log"
    model_dir= os.getcwd()+"/model_dir"

    os.makedirs(model_dir, exist_ok=True)

    # env = UnityEnvironment('Tennis_Windows_x86_64/Tennis.exe')
    env = UnityEnvironment('Tennis_Windows_x86_64/Tennis.exe', no_graphics = True)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)

    replay_episodes = 1000

    buffer = ReplayBuffer(int(replay_episodes * episode_length))

    # initialize policy and critic
    maddpg = MADDPG()
    # logger = SummaryWriter(log_dir=log_path)
    agent0_reward = []
    agent1_reward = []

    # training loop
    scores_deque = deque(maxlen=100)
    scores = []

    for episode in range(0, number_of_episodes):

        reward_this_episode = np.zeros( num_agents)
        env_info = env.reset(True)[brain_name]
        state = env_info.vector_observations

        obs = [[state[0], state[1]]]
        obs_full = np.concatenate((state[0], state[1]))

        #for calculating rewards for this particular episode - addition of all time steps

        frames = []
        tmax = 0

        for episode_t in range(episode_length):

            t += 1

            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            actions = maddpg.act(transpose_to_tensor(obs), noise=noise)
            noise *= noise_reduction

            actions_array = torch.stack(actions).detach().numpy()

            # transpose the list of list
            # flip the first two indices
            # input to step requires the first index to correspond to number of parallel agents
            # actions_for_env = np.rollaxis(actions_array,1)
            actions_for_env = np.clip(actions_array.flatten(), -1, 1)

            # print(actions_for_env)

            # step forward one frame
            # next_obs, next_obs_full, rewards, dones, info = env.step(actions_for_env)

            env_info = env.step(actions_for_env)[brain_name]
            next_state = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            next_obs = [[next_state[0], next_state[1]]]
            next_obs_full = np.concatenate((next_state[0], next_state[1]))

            # print(obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones


            # add data to buffer
            transition = ([obs],[obs_full], [actions_for_env], [rewards], [next_obs], [next_obs_full], [dones])

            buffer.push(transition)

            reward_this_episode += rewards

            obs, obs_full = next_obs, next_obs_full

            if any(dones):
                break

        # update once after every episode_per_update
        if len(buffer) > batchsize and episode % episode_per_update == 0:
            for a_i in range(num_agents):
                samples = buffer.sample(batchsize)
                maddpg.update(samples, a_i)
            maddpg.update_targets() #soft update the target network towards the actual networks

        avg_rewards = np.mean(reward_this_episode, axis=0)
        episode_reward = np.max(avg_rewards)
        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        print('\rEpisode {}\tAverage Score: {:.3f}\tEpisode Score: {:.3f}'.format(episode, np.mean(scores_deque), episode_reward), end="")

        if (episode > 0 and episode % 100 == 0) or episode == number_of_episodes-1:
            print('\rEpisode {}\tAverage Score: {:.3f}\tEpisode Score: {:.3f}'.format(episode, np.mean(scores_deque), episode_reward))

        if np.mean(scores_deque) >= 0.5:
            print('\nSuccess!')
            break

    #saving model
    save_dict_list =[]
    for i in range(num_agents):

        save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                        'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                        'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                        'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
        save_dict_list.append(save_dict)

        torch.save(save_dict_list,
                    os.path.join(model_dir, 'episode-{}.pt'.format(episode)))

    env.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.savefig('tennis_score_history.png')

    return scores

if __name__=='__main__':
    scores = main()
