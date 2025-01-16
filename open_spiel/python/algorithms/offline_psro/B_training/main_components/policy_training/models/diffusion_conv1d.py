""" This file is dedicated to implementing a diffusion model that is trained to generate trajectories to augment the offline dataset. """

import math
import numpy as np 
import tensorflow.compat.v1 as tf

class DiffusionModel(tf.Module):
    def __init__(self, input_shape, timesteps, betas, session, learning_rate):
        """ 
        input_shape: (sequence_length, state_size + action_indicator)
        timesteps: integer
        betas: list of length timesteps representing forward variances
        """
        self._input_shape = input_shape
        self._timesteps = timesteps 
        self._betas = betas
        self._sigmas = [np.sqrt(b) for b in self._betas]
        self._alphas = [1.0 - b for b in self._betas]
        self._alpha_bars = [np.prod(alphas[0:t+1]) for t, _ in enumerate(alphas)]
        self._is_frozen = False

        self._eps_ph = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32, name="eps_ph")
        self._time_ph = tf.placeholder(shape=None, dtype=tf.float32, name="time_ph")
        self._trajectory_ph = tf.placeholder(shape=[None]+input_shape, dtype=tf.float32, name="trajectory_ph")

        time_one_hot_vectors = tf.one_hot(self._time_ph, depth=timesteps, axis=-1)  # [N, self._timesteps]
        time_one_hot_vectors = tf.repeat(time_one_hot_vectors, repeats=self._input_shape[0], axis=1) # [N, seq_length, state_size + action_indicator + self._timesteps]

        network_input = tf.concat([self._trajectory_ph, time_one_hot_vectors], axis=2)  # [N, seq_length, state_size + action_indicator + self._timesteps]

        self._predictions = 

        self._loss = tf.reduce_mean(tf.reduce_sum(tf.square(self._eps_ph - self._predictions), axis=[1,2]), axis=0)

        self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self._learn_step = self._optimizer.minimize(self._loss)

        if not self._is_frozen:
            self._initialize(session)

    def get_random_normal(self, N):
        return tf.random.normal([N] + self._input_shape, mean=0.0, stddev=1.0)

    def generate_trajectory(self, session):
        x_t = self.get_random_normal(1)
        z = self._get_random_normal(self._timesteps - 1)
        for t in range(self._timesteps, 0, -1):
            z_t = z[t-1] if t > 1 else 0
            diff = x_t - ((1 - self._alphas[t]) / ((1 - self._alpha_bars[t]) ** (.5))) * self.forward(x_t, t)
            x_t_prime = (1 / (self._alphas[t] ** (.5))) * diff + self._sigmas[t] * z_t
            x_t = x_t_prime
        return x_t

    def _initialize(self):
    
    def forward(self):

    def broadcast(self):

    def update(self):

    def train(self, dataset, steps, batch_size, session):
        """
        dataset is of size [N, input_shape]
        """
        num_datapoints = len(dataset)
        timestep_list = list(range(1, self._timesteps+1))
        losses = []
        for i in range(steps):
            indices = np.random.choice(num_datapoints, size=batch_size)
            batch = np.take(dataset, indices)  # [batch, seq_length, depth]
            t = np.random.choice(timestep_list, size=batch_size)
            eps = self.get_random_normal(batch_size)
            loss = self.update(batch, eps, t)
            losses.append(loss)
        return losses




        