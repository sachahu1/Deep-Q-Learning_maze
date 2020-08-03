############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import collections

episode_length = 1000
batch_size = 100
buffer_size = 1000000
end_episode = True

buf_eps = 0.001
buf_alpha = 0.7

dqn_bellman = "advanced"
dqn_update_rate = 100
dqn_gamma = 0.9

agent_type = "epsilon greedy with delta"
agent_delta = 0.00005
epsilon_clip = 0

class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        self.episode_length = episode_length
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        self.discrete_action = None

        self.reach_counter = 0

        # Define Agent
        self.type = agent_type
        if self.type == "epsilon greedy with delta":
            self.epsilon = 1
        else:
            self.epsilon = agent_epsilon
        self.delta = agent_delta

        self.epsilon_clip = epsilon_clip

        # Create DQN
        self.dqn = DQN(bellman=dqn_bellman, gamma=dqn_gamma, update_rate=dqn_update_rate)
        self.Buffer = ReplayBuffer(buffer_size, batch_size)

        self.parameter_check(self.dqn)

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if (self.num_steps_taken) % self.dqn.update_rate == 0:
            self.dqn.update_target()
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        Q_values = self.dqn.q_network.forward(torch.tensor(state).float()).detach().numpy()
        action = np.argmax(Q_values)
        action = self.step(action=action)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action

        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = 0
        if (self.state[0] == next_state[0]) and (self.state[1] == next_state[1]):
            reward = -0.8*distance_to_goal

        if next_state[0] > self.state[0]:
            reward = 1 - 0.1*distance_to_goal

        if distance_to_goal <= 0.1:
            reward = 1.5 - distance_to_goal

        if distance_to_goal <= 0.05:
            reward = 2 - distance_to_goal

        if end_episode == True:
            if distance_to_goal<0.03:
                self.num_steps_taken = self.episode_length
                print("reached goal : ", self.reach_counter)
                self.reach_counter += 1

                if self.type == "greedy":
                    self.dqn.optimizer = torch.optim.Adam(self.dqn.q_network.parameters(), lr=0)
                    self.epsilon = 0
                    self.delta = 0
                    # print("greedy policy successful, stop training")
                    return
                else:
                    self.type = "greedy"
                    self.episode_length = 100
                    self.epsilon_save = self.epsilon
                    self.epsilon = 0
                    # print("testing greedy policy")
                return

            else:
                if (self.num_steps_taken % self.episode_length)==0 and (self.type == "greedy"): 
                    self.episode_length = 1000
                    self.num_steps_taken = self.episode_length
                    self.type = "epsilon greedy with delta"
                    self.epsilon = self.epsilon_save
                    # print("revert to training")
                    return

        if self.type != "greedy":
            # Create a transition
            transition = (self.state, self.discrete_action, reward, next_state)
            # Save transition
            self.Buffer.save_transition(transition)
            # If the buffer contains enough data to sample a batch
            if (len(self.Buffer.buf)>=self.Buffer.batch_size):
                batch, chosen_indexes = self.Buffer.sample_batch()
                loss = self.dqn.train_q_network(batch)
                self.Buffer.update_weight(chosen_indexes, self.dqn.TD_delta)


    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        self.type = "greedy"
        self.epsilon = 0
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        Q_values = self.dqn.q_network.forward(torch.tensor(state).float())
        Q_values = Q_values.detach().numpy()
        action = np.argmax(Q_values)
        action = self.step(action=action)
        return action

    # Function to make the agent take one step in the environment.
    def step(self, action = None):
        if self.type == "epsilon greedy with delta":
            actions = [0, 1, 2]
            p = np.ones(3)*((self.epsilon)/3)
            p[action] = 1-self.epsilon + self.epsilon/3
            discrete_action = np.random.choice(actions, p=p)
            if self.epsilon - self.delta > self.epsilon_clip:
                self.epsilon = self.epsilon - self.delta
            else : 
                self.epsilon = self.epsilon_clip
        elif self.type == "greedy":
            discrete_action = action
        else:
            print("What are you trying to achieve...")
            return

        self.discrete_action = discrete_action
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        return continuous_action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:  # Move right
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        if discrete_action == 1:  # Move up
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        if discrete_action == 2:  # Move down
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        return continuous_action


    def parameter_check(agent, dqn ):
        Agent_types = ["greedy", "epsilon greedy", "training", "epsilon greedy with delta"]
        Bellman_types = ["basic", "advanced"]
        r = 1

        print("--------------------------------------------------------------------------------")
        if agent.type not in Agent_types:
            print("error, check agent type\nCurrent agent: ",agent.type, "\nAccepted types: ", Agent_types)
            r = -1
        elif dqn.bellman_equation not in Bellman_types:
            print("error, check bellman equation\nCurrent bellman: ", dqn.bellman_equation,"\nAccepted bellman: ", Bellman_types)
            r = -1
        else:
            print("Training configuration : \n- Bellman : ", dqn.bellman_equation)
            if dqn.bellman_equation != "basic":
                print( "- Gamma : ",dqn.gamma)
                print("- Update target network every ", dqn.update_rate, "steps\n")
            else:
                print()
            print("Agent configuration : \n- type : ", agent.type)
            if agent.type == "epsilon greedy":
                print("- Epsilon : ", agent.epsilon)
            elif agent.type == "epsilon greedy with delta":
                print("- Epsilon : ",agent.epsilon)
                print("- Delta : ",agent.delta)
            else:
                print()
        print()
        return r

###################################################################################################################################
# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self, bellman = "basic", gamma=0.9, update_rate=20):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=3)
        # Create target network
        self.target_network = Network(input_dimension=2, output_dimension=3)
        self.update_rate = update_rate

        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        # Define bellman equation
        self.bellman_equation = bellman
        # Define gamma value.
        self.gamma = gamma

        self.TD_delta = 0

    def update_target(self):
        if self.bellman_equation == "advanced":
            self.target_network.load_state_dict(self.q_network.state_dict())

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, batch):
        if self.bellman_equation == "advanced":
            pass
        else:
            print("No bellman equation")
            return

        states = np.array([batch[i][0] for i in range(len(batch))])
        action = np.array([batch[i][1] for i in range(len(batch))])
        reward = np.array([batch[i][2] for i in range(len(batch))])
        next_states = np.array([batch[i][3] for i in range(len(batch))])

        states = torch.tensor(states)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_states = torch.tensor(next_states)

        reward = torch.unsqueeze(reward.float(),1)
        action = torch.unsqueeze(action.long(), 1)

        network_prediction = self.q_network.forward(states)

        Q_values = torch.gather(network_prediction, 1, action)

        Q_target = self.target_network.forward(next_states).detach()

        best_target_action = torch.argmax(Q_target, dim=1)
        Q_target_state = torch.max(Q_target, dim=1)[0]

        #### Compute Weight loss here ####
        self.TD_delta = (torch.unsqueeze(Q_target_state,1) - Q_values)
        self.TD_delta = torch.flatten(self.TD_delta).tolist()

        bellman_loss = torch.nn.MSELoss()(reward + torch.unsqueeze(Q_target_state, 1) * self.gamma, Q_values)
    
        return bellman_loss

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.eps = buf_eps
        self.buf = collections.deque(maxlen=self.buffer_size)
        self.Weights = collections.deque(maxlen=self.buffer_size)
        self.alpha = buf_alpha

    def save_transition(self, transition):
        self.buf.append(transition)
        if len(self.Weights) < self.batch_size:
            self.Weights.append(1/self.batch_size)
        else : 
            self.Weights.append(max(self.Weights))
        return

    def update_weight(self, chosen_indexes, TD_delta):
        for i in range(len(chosen_indexes)):
            self.Weights[chosen_indexes[i]] = abs(TD_delta[i]) + self.eps

    def compute_probabilities(self):
        probabilities = []
        ws = 0
        for i in range(len(self.Weights)):
            ws += self.Weights[i]**self.alpha

        for i in range(len(self.Weights)):
            probabilities.append((self.Weights[i]**self.alpha) / ws)
        return probabilities

    def sample_batch(self):    
        chosen_indexes = np.arange(len(self.Weights))
        chosen_indexes = np.random.choice(chosen_indexes, self.batch_size, p=self.compute_probabilities())

        batch = []
        for i in chosen_indexes :
            batch.append(self.buf[i])

        return batch, chosen_indexes

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output