from udacitysoccer import config
from udacitysoccer.models import Actor
from udacitysoccer.models import Critic
from common import *
from torch import optim

from udacitysoccer.support import Experience
from udacitysoccer.support import OUNoise
from udacitysoccer.support import ReplayBuffer


SIDES = ["goalie", "striker"]


def minimize_loss(loss, optimizer: optim.Adam, is_critic=False, critic=None):
    optimizer.zero_grad()
    loss.backward()
    if is_critic and critic is not None:
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
    optimizer.step()


def soft_update(local_model, target_model):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(config.TAU * local_param.data + (1.0 - config.TAU) * target_param.data)


class MultiSoccerAgent:
    def __init__(self, soccer_agents: list):
        self.soccer_agents = {}
        for soccer_agent in soccer_agents:
            self.soccer_agents[soccer_agent.side] = soccer_agent

    def reset(self):
        for side in self.soccer_agents:
            soccer_agent = self.soccer_agents[side]
            soccer_agent.reset()

    def act(self, states, side, add_noise=True):
        return self.soccer_agents[side].act(states, add_noise)

    def step(self, experience: Experience, side):
        self.soccer_agents[side].step(experience)

    def local_actor_network(self, side):
        return self.soccer_agents[side].local_actor_network()


class SoccerAgent:
    def __init__(self, side: str, state_size: int, action_size: int, num_agents: int, noise: OUNoise,
                 memory: ReplayBuffer):
        """
        Initialise object
        :param state_size: Dimension of the state space
        :param action_size: Dimension of the action space
        :param num_agents: The number of agents in the environment
        :param noise: Exploration noise
        :param memory: The replay buffer that stores the experiences for the benefit of experience replay
        """
        self.side = side
        self.action_size = action_size
        self.num_agents = num_agents
        self.agents = []
        self.noise = noise
        self.actor_local = Actor(name="{} Actor: Local".format(side.title()), state_size=state_size,
                                 action_size=action_size, random_seed=0).to(DEVICE)
        self.memory = memory
        self.ready_to_learn = len(self.memory) > config.BATCH_SIZE

        # Initialise agents
        for agent_id in range(self.num_agents):
            agent = Agent(side, agent_id, state_size, action_size, 0, self.actor_local, self.noise)
            self.agents.append(agent)

    def reset(self):
        self.noise.reset()

    def act(self, states, add_noise=True):
        """ Act based on the observed state
            :param add_noise: Flag to decide whether to add noise or not
            :param states: Observed state, which contains states of each of the agents
        """
        actions = np.zeros((self.num_agents, self.action_size))
        for agent in self.agents:
            agent_id = agent.agent_id
            actions[agent_id] += agent.act(state=states[agent_id], add_noise=add_noise)
        return np.clip(actions, -1, 1)

    def step(self, experience: Experience):
        """ Add experiences to the experience buffer and learn from a batch """
        self.memory.add(experience)
        if not self.ready_to_learn:
            self.ready_to_learn = len(self.memory) >= config.BATCH_SIZE
        else:
            experiences = self.memory.sample()  # Sample once or sample for each agent?
            for agent in self.agents:
                agent.learn(experiences)

    def local_actor_network(self):
        return self.actor_local


class Agent:
    def __init__(self, side: str, agent_id: int, state_size: int, action_size: int, random_seed, actor_local: Actor,
                 noise: OUNoise):
        """
        Initialise object
        :param agent_id: Identifier of an agent
        :param actor_local: The actor network
        :param noise: The exploration noise
        """
        self.agent_id = agent_id
        self.actor_local = actor_local
        self.noise = noise

        self.actor_target = Actor("{} Actor {}: Target".format(side.title(), agent_id), state_size, action_size, random_seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.ACTOR_LR)

        # Initialise the Critic networks (local and target)
        self.critic_local = Critic("{} Critic {}: Local".format(side.title(), agent_id), state_size, action_size, random_seed).to(DEVICE)
        self.critic_target = Critic("{} Critic {}: Target".format(side.title(), agent_id), state_size, action_size, random_seed).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.CRITIC_LR,
                                           weight_decay=config.WEIGHT_DECAY)

    def act(self, state, add_noise=True):
        """ Return the action for the state as per the policy """
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()  # put the policy in evaluation mode
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()  # put policy back in training mode
        if add_noise:
            action += self.noise.sample()
        return action

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        q_targets = rewards + (config.GAMMA * q_targets_next * (1 - dones))
        # compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        minimize_loss(critic_loss, self.critic_optimizer, is_critic=True, critic=self.critic_local)

        # update the actor
        actions_predicted = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_predicted).mean()
        minimize_loss(actor_loss, self.actor_optimizer)

        # update target networks
        soft_update(self.critic_local, self.critic_target)
        soft_update(self.actor_local, self.actor_target)

