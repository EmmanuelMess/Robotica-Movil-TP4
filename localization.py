""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import minimized_angle, plot_field, plot_robot, plot_path
from soccer_field import Field
import policies
from ekf import ExtendedKalmanFilter
from pf import ParticleFilter


def localize(env, policy, filt, x0, num_steps, plot=False):
    # Collect data from an entire rollout
    states_noisefree, states_real, action_noisefree, obs_noisefree, obs_real = \
            env.rollout(x0, policy, num_steps)
    states_filter = np.zeros(states_real.shape)
    states_filter[0, :] = x0.ravel()

    errors = np.zeros((num_steps, 3))
    position_errors = np.zeros(num_steps)
    mahalanobis_errors = np.zeros(num_steps)

    if plot:
        fig = env.get_figure()

    for i in range(num_steps):
        x_real = states_real[i+1, :].reshape((-1, 1))
        u_noisefree = action_noisefree[i, :].reshape((-1, 1))
        z_real = obs_real[i, :].reshape((-1, 1))
        marker_id = env.get_marker_id(i)

        if filt is None:
            mean, cov = x_real, np.eye(3)
        else:
            # filters only know the action and observation
            mean, cov = filt.update(env, u_noisefree, z_real, marker_id)
        states_filter[i+1, :] = mean.ravel()

        if plot:
            fig.clear()
            plot_field(env, marker_id)
            plot_robot(env, x_real, z_real)
            plot_path(env, states_noisefree[:i+1, :], 'g', 0.5)
            plot_path(env, states_real[:i+1, :], 'b')
            if filt is not None:
                filt.show(env, marker_id)
                plot_path(env, states_filter[:i+1, :2], 'r')
            fig.canvas.flush_events()

        errors[i, :] = (mean - x_real).ravel()
        errors[i, 2] = minimized_angle(errors[i, 2])
        position_errors[i] = np.linalg.norm(errors[i, :2])

        cond_number = np.linalg.cond(cov)
        if cond_number > 1e12:
            print('Badly conditioned cov (setting to identity):', cond_number)
            print(cov)
            cov = np.eye(3)
        mahalanobis_errors[i] = \
                errors[i:i+1, :].dot(np.linalg.inv(cov)).dot(errors[i:i+1, :].T)

    mean_position_error = position_errors.mean()
    mean_mahalanobis_error = mahalanobis_errors.mean()
    anees = mean_mahalanobis_error / 3

    if filt is not None:
        print('-' * 80)
        print('Mean position error:', mean_position_error)
        print('Mean Mahalanobis error:', mean_mahalanobis_error)
        print('ANEES:', anees)

    if plot:
        plt.show(block=True)

    return mean_position_error, mean_mahalanobis_error, anees


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filter_type', choices=('none', 'ekf', 'pf'),
        help='filter to use for localization')
    parser.add_argument(
        '--plot', action='store_true',
        help='turn on plotting')
    parser.add_argument(
        '--seed', type=int,
        help='random seed')
    parser.add_argument(
        '--num-steps', type=int, default=200,
        help='timesteps to simulate')

    # Noise scaling factors
    parser.add_argument(
        '--data-factor', type=float, default=1,
        help='scaling factor for motion and observation noise (data)')
    parser.add_argument(
        '--filter-factor', type=float, default=1,
        help='scaling factor for motion and observation noise (filter)')
    parser.add_argument(
        '--num-particles', type=int, default=100,
        help='number of particles (particle filter only)')
    parser.add_argument(
        '--plot-error', action='store_true',
        help='plot error varying alpha and beta')

    return parser


def start(data_factor, filter_factor, num_particles, plot, args):
    alphas = np.array([0.05 ** 2, 0.005 ** 2, 0.1 ** 2, 0.01 ** 2])
    beta = np.diag([np.deg2rad(5) ** 2])

    env = Field(data_factor * alphas, data_factor * beta)
    policy = policies.OpenLoopRectanglePolicy()

    initial_mean = np.array([180, 50, 0]).reshape((-1, 1))
    initial_cov = np.diag([10, 10, 1])

    if args.filter_type == 'none':
        filt = None
    elif args.filter_type == 'ekf':
        filt = ExtendedKalmanFilter(
            initial_mean,
            initial_cov,
            filter_factor * alphas,
            filter_factor * beta
        )
    elif args.filter_type == 'pf':
        filt = ParticleFilter(
            initial_mean,
            initial_cov,
            num_particles,
            filter_factor * alphas,
            filter_factor * beta
        )

    # You may want to edit this line to run multiple localization experiments.
    return localize(env, policy, filt, initial_mean, args.num_steps, plot)


if __name__ == '__main__':
    args = setup_parser().parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if not args.plot_error:
        print('Data factor:', args.data_factor)
        print('Filter factor:', args.filter_factor)

        _ = start(args.data_factor, args.filter_factor, args.num_particles, args.plot, args)
    else:
        if args.plot:
            print("--plot ignored!")

        AVERAGING_RUNS = 10
        RS = [1 / 64, 1 / 16, 1 / 4, 4, 16, 64]
        PARTICLES = [20, 50, 100, 500]

        if args.filter_type == 'pf':
            mean_position_errors = [[] for _ in PARTICLES]
            mean_mahalanobis_errors = [[] for _ in PARTICLES]
            aneess = [[] for _ in PARTICLES]

            for r in RS:
                for i, num_particles in enumerate(PARTICLES):
                    mean_position_runs = []
                    mean_mahalanobis_runs = []
                    anees_runs = []

                    for _ in range(AVERAGING_RUNS):
                        mean_position_error, mean_mahalanobis_error, anees = start(r, r, num_particles, False, args)

                        mean_position_runs.append(mean_position_error)
                        mean_mahalanobis_runs.append(mean_mahalanobis_error)
                        anees_runs.append(anees)

                    mean_position_errors[i].append(np.mean(mean_position_runs))
                    mean_mahalanobis_errors[i].append(np.mean(mean_mahalanobis_runs))
                    aneess[i].append(np.mean(anees_runs))

            class _:
                plt.title("Error de posicion medio sobre valores de r")
                plt.xlabel('r')
                plt.ylabel('Error')
                plt.gca().set_xscale('log')
                plt.gca().set_yscale('log')
                plt.plot(RS, mean_position_errors[PARTICLES.index(100)], label=f"Error de posicion medio")
                plt.savefig('plots/pf-b.png')
                plt.clf()

                plt.title("Error de posicion medio y ANEES sobre valores de r")
                plt.xlabel('r')
                plt.ylabel('Error')
                plt.gca().set_xscale('log')
                plt.gca().set_yscale('log')
                plt.plot(RS, mean_position_errors[PARTICLES.index(100)], label=f"Error de posicion medio")
                plt.plot(RS, aneess[PARTICLES.index(100)], label=f"ANEES")
                plt.legend()
                plt.savefig('plots/pf-c.png')
                plt.clf()

                plt.title("Error de posicion medio y ANEES sobre valores de r")
                plt.xlabel('r')
                plt.ylabel('Error')
                plt.gca().set_xscale('log')
                plt.gca().set_yscale('log')
                for i, y in enumerate(mean_position_errors):
                    if i == PARTICLES.index(100):
                        continue
                    plt.plot(RS, y, label=f"Error de posicion medio, {PARTICLES[i]} particulas")
                for i, y in enumerate(aneess):
                    if i == PARTICLES.index(100):
                        continue
                    plt.plot(RS, y, label=f"ANEES, {PARTICLES[i]} particulas")
                plt.legend()
                plt.savefig('plots/pf-d.png')
                plt.clf()

        else:
            mean_position_errors = []
            mean_mahalanobis_errors = []
            aneess = []
            for r in RS:
                mean_position_runs = []
                mean_mahalanobis_runs = []
                anees_runs = []

                for _ in range(AVERAGING_RUNS):
                    mean_position_error, mean_mahalanobis_error, anees = start(r, r, None, False, args)

                    mean_position_runs.append(mean_position_error)
                    mean_mahalanobis_runs.append(mean_mahalanobis_error)
                    anees_runs.append(anees)

                mean_position_errors.append(np.mean(mean_position_runs))
                mean_mahalanobis_errors.append(np.mean(mean_mahalanobis_runs))
                aneess.append(np.mean(anees_runs))

            class _:
                plt.title("Error de posicion medio sobre valores de r")
                plt.xlabel('r')
                plt.ylabel('Error')
                plt.gca().set_xscale('log')
                plt.gca().set_yscale('log')
                plt.plot(RS, mean_position_errors)
                plt.savefig('plots/ekf-b.png')
                plt.clf()

                plt.title("Error de posicion medio y ANEES sobre valores de r")
                plt.xlabel('r')
                plt.ylabel('Error')
                plt.gca().set_xscale('log')
                plt.gca().set_yscale('log')
                plt.plot(RS, mean_position_errors, label="Error de posicion medio")
                plt.plot(RS, aneess, label="ANEES")
                plt.legend()
                plt.savefig('plots/ekf-c.png')
                plt.clf()


