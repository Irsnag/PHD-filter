# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 20:11:38 2024

@author: 33606
"""

import numpy as np
from datetime import datetime, timedelta
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.plotter import Plotter
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

start_time = datetime.now()

start_time = datetime.now()

np.random.seed(1991)

q_x = 0.05
q_y = 0.05
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])

truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

measurement_model = LinearGaussian(
    ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0, 2),  # Mapping measurement vector index to state index
    noise_covar=np.array([[5, 0],  # Covariance matrix for Gaussian PDF
                          [0, 5]])
    )

measurements = []
predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
track = Track()

num_steps = 20

for k in range(1, num_steps + 1):
    
    state = truth[k-1]
    
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement,
                                  timestamp=state.timestamp,
                                  measurement_model=measurement_model))
    
    prediction = predictor.predict(prior, timestamp=measurements[k-1].timestamp)
    hypothesis = SingleHypothesis(prediction, measurements[k-1])  
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]
    
    truth.append(GroundTruthState(
        transition_model.function(state, noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))
    

plotter = Plotter()
plotter.plot_ground_truths(truth, [0, 2])
plotter.plot_measurements(measurements, [0, 2])
plotter.plot_tracks(track, [0, 2], uncertainty=True)
plotter.fig
    
                                            