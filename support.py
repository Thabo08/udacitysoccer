""" Support classes """
from collections import deque, namedtuple
from common import *
import copy


class Experience:
    """ Helper class to encapsulate an experience """

    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def get_state(self):
        return self.state

    def get_action(self):
        return self.action

    def get_reward(self):
        return self.reward

    def get_next_state(self):
        return self.next_state

    def get_done(self):
        return self.done


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, random_seed: int):
        """ Initialize a ReplayBuffer object.
            :param buffer_size (int): maximum size of buffer
            :param batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(random_seed)

    def add(self, experience: Experience):
        """Add a new experience to memory."""
        e = self.experience(experience.get_state(), experience.get_action(), experience.get_reward(),
                            experience.get_next_state(), experience.get_done())
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """ Ornstein-Uhlenbeck exploration noise process for temporally correlated noise """

    def __init__(self, action_size, seed, mu=0., theta=.15, sigma=.2):
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.action_size = action_size
        random.seed(seed)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.action_size)
        self.state = x + dx
        return self.state
