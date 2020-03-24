""" the Tracker class that handles tracking using filter

Author: Henry Zhang
Date:August 23, 2019
"""

# module
import math
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3)

from kalman_filter import KalmanFilter
from utils import gate_probability
# parameters


# classes
class Estimation():
  def __init__(self, obj_id, state, variance, innovation):
    self.id = obj_id
    self.state = state
    self.variance = variance
    self.innovation = innovation
  
  def is_close(self, estimate):
    state_diff = self.state - estimate.state
    diff_norm = np.linalg.norm(state_diff)
    print("diff_norm", diff_norm, self.id, estimate.id, state_diff)
    print(self.state, estimate.state)
    return diff_norm < 2

class Tracker():
  def __init__(self, time, observation, id=0):
    self.filter = self.initialize(time, observation)
    self.observation = observation
    self.id = 0
    self.estimate = Estimation(self.id, \
                               self.filter.x_post, \
                               self.filter.P_post, \
                               self.filter.innovation)
    self.threshold_associate = 2
  
  def initialize(self, time, observation):
    self.update_time = time
    self.predict_time = time
    noise_observation = np.eye(observation.shape[0])
    noise_process = np.eye(observation.shape[0])
    mtx_transition = np.eye(observation.shape[0])
    mtx_observation = np.eye(observation.shape[0])
    mtx_control = None
    filter_ = KalmanFilter(x_prior=observation, 
                           P_prior=noise_observation, 
                           mtx_transition=mtx_transition,
                           mtx_observation=mtx_observation,
                           mtx_control=mtx_control,
                           noise_process=noise_process,
                           noise_observation=noise_observation
                           )
    return filter_

  def predict(self, time_acc):
    self.filter.predict(time_acc - self.predict_time)
    self.estimate = Estimation(self.id, \
                               self.filter.x_pre, \
                               self.filter.P_pre, \
                               self.filter.innovation)
    self.predict_time = time_acc
  
  def update(self, observation):
    self.filter.update(observation)
    self.estimate = Estimation(self.id, \
                                self.filter.x_post, \
                                self.filter.P_post, \
                                self.filter.innovation)
    self.update_time = self.predict_time
    self.observation = observation

    # check probability
    prob = gate_probability(self.filter.innovation, self.filter.innovation_cov)
    print(prob, self.filter.innovation.T)
    
# functions


# main
def main():
  pass

if __name__ == "__main__":
  main()