""" This file holds the implementation of the Actor (policy function) and the Critic (value function) models used
 in the DDPG algorithm """

import abc

from common import *


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


def reset_parameters(layers):
    for ith_layer in range(0, len(layers) - 1):
        layer = layers[ith_layer]
        layer.weight.data.uniform_(*hidden_init(layer))
    layers[-1].weight.data.uniform_(-3e-3, 3e-3)


class Model(nn.Module):
    """ The abstract Model """
    def __init__(self, name, state_size, action_size, random_seed, *args):
        """ Initialise model parameters

            :param name: Specifies the name of the model (for convenience)
            :param state_size: Dimension of the state space of an environment
            :param action_size: Dimension of the action space of an environment
            :param random_seed: Random seed
            :param args: Sizes of hidden layers
         """
        if len(args) == 0:
            raise ValueError("Hidden layer units not specified")
        super(Model, self).__init__()
        torch.manual_seed(random_seed)
        self.state_size = state_size
        self.action_size = action_size
        self.name = name

    @abc.abstractmethod
    def forward(self, state, action=None):
        pass

    def print_(self):
        print("Initialised '{}' model".format(self.name))


class Actor(Model):
    def __init__(self, name, state_size, action_size, random_seed, fc1_units=256, fc2_units=128):
        """
        Initialise Actor model (policy gradient function)
        :param fc1_units: Nodes in 1st hidden layer
        :param fc2_units: Nodes in 2nd hidden layer
        """
        super().__init__(name, state_size, action_size, random_seed, fc1_units, fc2_units)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        reset_parameters([self.fc1, self.fc2, self.fc3])
        self.print_()

    def forward(self, state, action=None):
        """ Perform forward pass and map state to action """
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        probs = F.softmax(self.fc3(state), dim=0)
        distribution = Categorical(probs)
        return distribution.sample()


class Critic(Model):
    def __init__(self, name, state_size, action_size, random_seed, fc1_units=256, fc2_units=128):
        """
        Initialise Critic model (value based function)
        :param fc1_units: Nodes in 1st hidden layer
        :param fc2_units: Nodes in 2nd hidden layer
        """
        super().__init__(name, state_size, action_size, random_seed, fc1_units, fc2_units)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        reset_parameters([self.fc1, self.fc2, self.fc3])
        self.print_()

    def forward(self, state, action=None):
        """ Perform forward pass and map state and action to Q values """
        assert action is not None, "Action cannot be none"
        xs = F.leaky_relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)

    # def forward(self, state):
    #     x = F.relu(self.fc1(state))
    #     x = F.relu(self.fc2(x))
    #
    #     value = self.fc_critic(x)
    #
    #     return value
