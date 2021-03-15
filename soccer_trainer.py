import datetime
from collections import deque

import numpy as np
import torch
from unityagents import UnityEnvironment

from udacitysoccer import config
from udacitysoccer.agents import MultiSoccerAgent
from udacitysoccer.agents import SoccerAgent
from udacitysoccer.support import Experience
from udacitysoccer.support import OUNoise
from udacitysoccer.support import ReplayBuffer


def env_settings():
    settings = {}
    env = UnityEnvironment(file_name="Soccer.app")
    settings["env"] = env

    # print the brain names
    print(env.brain_names)

    for i in range(2):
        brain_name = env.brain_names[i]
        brain = env.brains[brain_name]
        print(brain_name)
        # settings[brain_name] = brain_name

        classifier = "goalie" if i == 0 else "striker"

        # reset the environment
        env_info = env.reset(train_mode=True)

        # number of agents
        num_agents = len(env_info[brain_name].agents)
        print('Number of {} agents:'.format(classifier), num_agents)
        settings["num_{}_agents".format(classifier)] = num_agents

        action_size = brain.vector_action_space_size  # vector_action_space_size
        print('Number of {} actions:'.format(classifier), action_size)
        settings["num_{}_actions".format(classifier)] = action_size

        # examine the state space
        states = env_info[brain_name].vector_observations
        state_size = states.shape[1]
        settings["{}_states".format(classifier)] = states
        settings["num_{}_states".format(classifier)] = state_size
    settings['brain_names'] = env.brain_names
    return settings


def random_steps(settings: dict):
    env = settings["env"]
    g_brain_name = env.brain_names[0]
    s_brain_name = env.brain_names[1]
    num_g_agents = settings["num_goalie_agents"]
    num_s_agents = settings["num_striker_agents"]
    g_action_size = settings["num_goalie_actions"]
    s_action_size = settings["num_striker_actions"]
    for i in range(2):  # play game for 2 episodes
        env_info = env.reset(train_mode=False)  # reset the environment
        g_states = env_info[g_brain_name].vector_observations  # get initial state (goalies)
        s_states = env_info[s_brain_name].vector_observations  # get initial state (strikers)
        g_scores = np.zeros(num_g_agents)  # initialize the score (goalies)
        s_scores = np.zeros(num_s_agents)  # initialize the score (strikers)
        while True:
            # select actions and send to environment
            g_actions = np.random.randint(g_action_size, size=num_g_agents)
            s_actions = np.random.randint(s_action_size, size=num_s_agents)
            # g_actions = np.random.randint(g_action_size, size=g_action_size)
            # s_actions = np.random.randint(s_action_size, size=s_action_size)
            actions = dict(zip([g_brain_name, s_brain_name],
                               [g_actions, s_actions]))
            env_info = env.step(actions)

            # get next states
            g_next_states = env_info[g_brain_name].vector_observations
            s_next_states = env_info[s_brain_name].vector_observations

            # get reward and update scores
            g_rewards = env_info[g_brain_name].rewards
            s_rewards = env_info[s_brain_name].rewards
            g_scores += g_rewards
            s_scores += s_rewards

            # check if episode finished
            done = np.any(env_info[g_brain_name].local_done)

            # roll over states to next time step
            g_states = g_next_states
            s_states = s_next_states

            # exit loop if episode finished
            if done:
                break
        print('Scores from episode {}: {} (goalies), {} (strikers)'.format(i + 1, g_scores, s_scores))


def step_tuple(env_info, brain_name):
    """ Returns a tuple of next state, reward, and done when the agent steps through the environment based
        on the action taken
        :param brain_name: The brain name to get the experience for
        :param env_info: Object holding information about the environment at a certain point
    """
    return env_info[brain_name].vector_observations, env_info[brain_name].rewards, env_info[brain_name].local_done


def ddpg(agent: MultiSoccerAgent, env_settings: dict, num_episodes=2000, max_time_steps=1000, target=1.0):
    """ Train an agent using the DDPG algorithm

        :param env_settings: Settings of the environment
        :param agent: a continuous control agent
        :param num_episodes: the number of episodes to train the agent
        :param target: The average target score the agent needs to achieve for optimal performance
        :param max_time_steps: Maximum time steps per episode
    """
    now = datetime.datetime.now()
    print(now, "- Training a multi agent for max {} episodes. Target score to reach is {}".format(num_episodes, target))
    # collections to help keep track of the score
    scores_deque = deque(maxlen=100)
    scores = []
    stats = {"scores": [], "episodes": []}  # collects stats for plotting purposes
    mean_score = 0.
    env = env_settings["env"]
    brain_names = settings["brain_names"]
    num_goalie_agents = env_settings["num_goalie_agents"]
    num_striker_agents = env_settings["num_striker_agents"]
    saved_model = 'checkpoint_{}.pth'

    for episode in range(1, num_episodes + 1):
        goalie_env_info, striker_env_info = tuple(env.reset(train_mode=True)[brain_name] for brain_name in brain_names)
        goalie_states, striker_states = goalie_env_info.vector_observations, striker_env_info.vector_observations
        agent.reset()
        score = np.zeros(num_striker_agents)

        for _ in range(max_time_steps):
            goalie_actions = agent.act(goalie_states, "goalie")[0]
            striker_actions = agent.act(striker_states, "striker")[0]

            # goalie_actions = np.random.randint(4, size=2)
            # striker_actions = np.random.randint(6, size=2)

            actions = dict(zip(brain_names, [goalie_actions, striker_actions]))
            env_info = env.step(actions)

            goalie_next_states, goalie_rewards, goalie_dones = step_tuple(env_info, brain_names[0])
            striker_next_states, striker_rewards, striker_dones = step_tuple(env_info, brain_names[1])
            for idx in range(num_goalie_agents):
                agent.step(Experience(goalie_states[idx], goalie_actions[idx], goalie_rewards[idx],
                                      goalie_next_states[idx], goalie_dones[idx]), "goalie")
            goalie_states = goalie_next_states
            for idx in range(num_striker_agents):
                agent.step(Experience(striker_states[idx], striker_actions[idx], striker_rewards[idx],
                                      striker_next_states[idx], striker_dones[idx]), "striker")
            striker_states = striker_next_states
            score += striker_rewards
            done = np.any(striker_dones) or np.any(goalie_dones)
            if done:
                break

        max_score = max(score)  # todo: verify this
        scores_deque.append(max_score)
        scores.append(max_score)
        mean_score = np.mean(scores_deque)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score))

        stats["scores"].append(max_score)
        stats["episodes"].append(episode)

        if mean_score >= target:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, mean_score))
            print("Target score of {0} has been reached. Saving model to {1}".format(target, saved_model))
            torch.save(agent.local_actor_network("goalie").state_dict(), saved_model.format("goalie"))
            torch.save(agent.local_actor_network("striker").state_dict(), saved_model.format("striker"))
            break

    now = datetime.datetime.now()
    print(now, "- Finished training " + "successfully!" if mean_score >= target else "unsuccessfully!")
    return scores, stats


if __name__ == '__main__':
    # random_steps()
    # env.close()
    settings = env_settings()
    run_random = False
    if run_random:
        random_steps(settings)
    else:
        sides = ["goalie", "striker"]
        soccer_agents = []
        for side in sides:
            state_size = settings["num_{}_states".format(side)]
            num_agents = settings["num_{}_agents".format(side)]
            action_size = settings["num_{}_actions".format(side)] // 2
            memory = ReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, random_seed=0)
            noise = OUNoise(action_size, 0)
            soccer_agent = SoccerAgent(side=side, state_size=state_size, action_size=action_size, num_agents=num_agents,
                                       noise=noise, memory=memory)

            soccer_agents.append(soccer_agent)

        multi_soccer_agent = MultiSoccerAgent(soccer_agents)
        _, stats = ddpg(agent=multi_soccer_agent, env_settings=settings)

    settings["env"].close()
