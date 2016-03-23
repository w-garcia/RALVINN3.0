import theano
import lasagne
import numpy as np
import theano.tensor as t


class DeepQLearner:

    """
    Creates a RL agent that learns to behave optimally.
    """

    def __init__(
            self,
            input_width,
            input_height,
            n_actions,
            discount,
            learn_rate,
            batch_size,
            rng
    ):

        self.input_width = input_width
        self.input_height = input_height
        self.n_actions = n_actions
        self.discount = discount
        self.lr = learn_rate
        self.batch_size = batch_size
        self.rng = rng

        lasagne.random.set_rng(self.rng)

        self.l_out = self.build_network(
            batch_size,
            input_width,
            input_height,
            n_actions
        )

        states = t.tensor4('states')
        next_states = t.tensor4('next_states')
        rewards = t.col('rewards')
        actions = t.icol('actions')
        terminals = t.icol('terminals')

        self.states_shared = theano.shared(
            np.zeros((batch_size, 1, input_height, input_width),
                     dtype=theano.config.floatX))

        self.next_states_shared = theano.shared(
            np.zeros((batch_size, 1, input_height, input_width),
                     dtype=theano.config.floatX))

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        q_vals = lasagne.layers.get_output(self.l_out, states)

        next_q_vals = lasagne.layers.get_output(self.l_out, next_states)
        next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        target = (rewards +
                  (t.ones_like(terminals) - terminals) *
                  self.discount * t.max(next_q_vals, axis=1, keepdims=True))
        diff = target - q_vals[t.arange(batch_size),
                               actions.reshape((-1,))].reshape((-1, 1))

        loss = t.sum(0.5 * diff ** 2)

        params = lasagne.layers.helper.get_all_params(self.l_out)
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }

        updates = lasagne.updates.sgd(loss, params, self.lr)

        self._train = theano.function([], [loss, q_vals], updates=updates,
                                      givens=givens)
        self._q_vals = theano.function([], q_vals,
                                       givens={states: self.states_shared})

    def build_network(self, batch_size, input_width, input_height, n_actions):

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, 1, input_width, input_height)
        )

        l_out = lasagne.layers.DenseLayer(
            l_in,
            num_units=n_actions,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def train(self, states, actions, rewards, next_states, terminals):

        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)

        loss, _ = self._train()

        return np.sqrt(loss)

    def q_vals(self, state):

        states = np.zeros((self.batch_size, 1, self.input_height,
                           self.input_width), dtype=theano.config.floatX)
        states[0, ...] = state
        self.states_shared.set_value(states)

        return self._q_vals()[0]

    def choose_action(self, state, epsilon):

        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.n_actions)
        q_vals = self.q_vals(state)

        return np.argmax(q_vals)

    def get_weights(self):

        """
        Gets the weights of the hidden layer for visualization and analysis.
        :return: weights of the hidden layer
        """

        weights = lasagne.layers.get_all_param_values(self.l_out)[0]
        return weights
