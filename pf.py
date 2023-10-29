""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np

from utils import minimized_angle


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """

        self.particles[:, 2] = np.mod(self.particles[:, 2] + np.pi + u[2], 2 * np.pi) - np.pi

        def update(theta):
            matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            return np.matmul(matrix, u[:2]).reshape((2,))

        rotatedU = np.array([update(theta) for theta in self.particles[:, 2]])

        self.particles[:, :2] = self.particles[:, :2] + rotatedU

        newWeights = []

        for particle in self.particles:
            zParticle = env.observe(particle, marker_id)
            weight = (-np.abs(z - zParticle)*100).reshape((1,))[0]
            newWeights.append(weight)

        print("Best match: ", self.particles[np.argmax(newWeights)])

        newWeights = self.weights * np.array(newWeights)
        W = np.sum(newWeights)

        self.weights = newWeights / W

        new_particles, _ = self.resample(self.particles, self.weights)

        print("Thinking: ", np.mean(new_particles, axis=0))

        mean, cov = self.mean_and_variance(new_particles)
        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        new_particles, new_weights = particles, weights

        new_particles = np.random.default_rng(seed=0).choice(particles, self.num_particles, replace=True, p=new_weights)

        return new_particles, new_weights

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.cos(particles[:, 2]).sum(),
            np.sin(particles[:, 2]).sum()
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles

        return mean.reshape((-1, 1)), cov
