""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas  # Alphas para modelar el noise
        self.beta = beta  # Qt

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov
    
    def _prediction_step(self,env, u):
        #________________Prediction Step____________________
        # self.mu es el mu en t-1, creo G y V (prev_theta del self.mu.ravel() te da el angulo en el t-1)
        Gt = env.G(self.mu, u)  # u = (rot1, rtrans, rot2)
        Vt = env.V(self.mu, u)  # u = (rot1, rtrans, rot2)
        def M(u,alphas):
            # M is M
            return env.noise_from_motion(u, alphas)
        Mt = M(u,self.alphas)  # Noise matrix 
        next_mu = env.forward(self.mu,u)  # Update mu, next_mu = mu + [rot_trans*cos(prev_theta+rot1),rot_trans*sin(prev_theta+rot1),rot1+rot2 ]
        next_sigma = (Gt@self.sigma@Gt.T) + (Vt@Mt@Vt.T)
        return next_mu,next_sigma

    def _correction_step(self, env, predicted_not_corrected_mu, predicted_not_corrected_sigma,
                         zt, features_observados_con_id):
        #________________Correction Step____________________
        predicted_corrected_carry_mu = predicted_not_corrected_mu
        predicted_corrected_carry_sigma = predicted_not_corrected_sigma
                             
        Qt = self.beta
        for (marker_id, landmark_observado) in features_observados_con_id:
            predicted_mu_x, predicted_mu_y, predicted_mu_theta = predicted_corrected_carry_mu.ravel()
            j = marker_id
            q = (env.MARKER_X_POS[j] - predicted_mu_x)**2 + (env.MARKER_Y_POS[j] - predicted_mu_y)**2
            z_piquito = np.array([[np.sqrt([q])[0], minimized_angle(np.arctan2([env.MARKER_Y_POS[j] - predicted_mu_y],
                                                                               [env.MARKER_X_POS[j] - predicted_mu_x])[0] - predicted_mu_theta)]])
            Ht = env.H(predicted_corrected_carry_mu,j)
            St = Ht@predicted_corrected_carry_sigma@Ht.T + Qt
            Kt = (predicted_corrected_carry_sigma@H.T)@np.linalg.pinv(St)
            predicted_corrected_carry_mu = predicted_corrected_carry_mu + Kt@(zt - z_piquito)
            predicted_corrected_carry_sigma = (np.eye(3) - Kt@Ht) @predicted_corrected_carry_sigma

        return predicted_corrected_carry_mu, predicted_corrected_carry_sigma

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID ("Feature observado", cantidad 1)
        """
        next_mu, next_sigma = self._prediction_step(env,u)  # Predicted mu and sigma (not corrected yet)
        self.mu, self.sigma = self._correction_step(env, next_mu, next_sigma, z, [(marker_id,z)])  # Correction over predicted mu and sigma 
        return self.mu, self.sigma
