#!/usr/bin/env python

# ROS library
import numpy as np


class KalmanFilter:
    """ A Kalman filter for pose estimation that merges odometry and photogrammetry. """
    def __init__(self):
        # covariance matrices
        self.mat_measure_cov = np.diag([1.0, 1.0, np.deg2rad(1.0)])     # uncertainty on photogrammetry
        self.mat_process_cov = np.diag([0.1, 0.1, np.deg2rad(1.0)])     # uncertainty on odometry
        self.mat_state_cov = 0.01*np.diag([1.0, 1.0, np.deg2rad(1.0)])  # uncertainty on state

        # state estimation [x y yaw]
        self.vec_xest = np.array([0.0, 0.0, 0.0])

        # model parameter
        # x_(k+1) = A*x_k + B*u_k + epsilon_k
        # y_k = C*x_k + gamma_k
        self.mat_ss_a = np.eye(3)
        self.mat_ss_b = np.eye(3)
        self.mat_ss_c = np.eye(3)

    def predict(self, u):
        """ Prediction step of the Kalman filter.

        :param u: odometry
        :return: robot's pose estimation after the prediction step
        """

        # a priori state estimate
        self.vec_xest = self.mat_ss_a.dot(self.vec_xest) + self.mat_ss_b.dot(u)

        # ensure angle is between -pi and pi
        self.vec_xest[2] = (self.vec_xest[2] + np.pi) % (2 * np.pi) - np.pi

        # a priori state covariance
        self.mat_state_cov = self.mat_ss_a.dot(self.mat_state_cov).dot(self.mat_ss_a.T) \
                             + self.mat_process_cov
        
        return self.vec_xest
        
    def update(self, y, pred, mat_state_cov): #, (mat_state_cov)
        """ Update step of the Kalman filter.

        :param y: photogrammetry
        :return: robot's pose estimation after the update step
        """
        # innovation
        vec_inition = y - self.mat_ss_c.dot(pred)

        # ensure angle is between -pi and pi
        vec_inition[2] = (vec_inition[2] + np.pi) % (2 * np.pi) - np.pi

        # innovation covariance
        #mat_inition_cov = self.mat_ss_c.dot(self.mat_state_cov).dot(self.mat_ss_c.T) \
        #                     + self.mat_measure_cov
        mat_inition_cov = self.mat_ss_c.dot(mat_state_cov).dot(self.mat_ss_c.T) \
                           + self.mat_measure_cov

        # kalman gain
        #mat_kalman_gain = self.mat_state_cov.dot(self.mat_ss_c.T).dot(np.linalg.inv(mat_inition_cov))
        mat_kalman_gain = mat_state_cov.dot(self.mat_ss_c.T).dot(np.linalg.inv(mat_inition_cov))

        # a posteriori state estimate
        self.vec_xest = pred + mat_kalman_gain.dot(vec_inition)

        # a posteriori state covariance
        self.mat_state_cov = (np.eye(len(self.vec_xest))-mat_kalman_gain.dot(self.mat_ss_c)).dot(self.mat_state_cov)

        return self.vec_xest
