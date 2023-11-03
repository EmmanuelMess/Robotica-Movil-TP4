""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np
import scipy

import utils
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

        def update(x):
            prev_x, prev_y, prev_theta = x
            rot1, trans, rot2 = u

            theta = prev_theta + rot1
            x_next = np.array([
                prev_x + trans * np.cos(theta),
                prev_y + trans * np.sin(theta),
                minimized_angle(theta + rot2)])

            return x_next

        self.particles[:] = np.array([update(x).reshape((-1,)) for x in self.particles[:]])

        newWeights = []

        for particle in self.particles:
            zParticle = env.observe(particle, marker_id)
            diff = utils.minimized_angle(z - zParticle)
            weight = np.float64(scipy.stats.norm.pdf(diff, loc=0, scale=2))
            newWeights.append(weight)

        newWeights = self.weights * np.array(newWeights)
        W = np.sum(newWeights)

        self.weights = newWeights / W

        self.particles, _ = self.resample(self.particles, self.weights)

        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        new_particles = np.random.default_rng(seed=0).choice(particles, self.num_particles, replace=True, p=weights)

        return new_particles, weights

    def show(self, env, marker_id):
        for particle in self.particles:
            zParticle = env.observe(particle, marker_id)
            self.plot_particle(env, particle, zParticle.reshape((1,)))

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

    def plot_particle(self, env, x, z, radius=5):
        """Plot the robot on the soccer field."""
        ax = env.get_figure().gca()
        utils.plot_circle(ax, x[:2], radius=radius, facecolor='r')

        # robot orientation
        ax.plot(
            [x[0], x[0] + np.cos(x[2]) * (radius + 5)],
            [x[1], x[1] + np.sin(x[2]) * (radius + 5)],
            'k')

        # observation
        ax.plot(
            [x[0], x[0] + np.cos(x[2] + z[0]) * 100],
            [x[1], x[1] + np.sin(x[2] + z[0]) * 100],
            'r', linewidth=0.5)