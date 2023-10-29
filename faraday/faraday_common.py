import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import numpy as np
import os
import math
import collections

class Observation:
    def __init__(self, mag_field_volt, mag_field_curr, filter, sample, min_trans_eye,
                angles, intensities):
        self.mag_field_volt = mag_field_volt
        self.mag_field_curr = mag_field_curr
        self.filter = filter
        self.sample = sample
        self.min_trans_eye = min_trans_eye
        self.angles = angles
        self.intensities = intensities

class MalusFit:
    def __init__(self, observation):
        self.observation = observation
        self.angles = observation.angles
        self.intensities = observation.intensities
        self.fit()
        
    def fit(self):
        """
        Fit malus curve to angles and intensities
        """
        m1_guess = np.max(self.intensities)
        m2_guess = np.max(self.intensities)
        m3_guess = self.angles[np.argmin(self.intensities)]
        params, covariance = curve_fit(self.malus_curve, 
                                           self.angles, 
                                           self.intensities, 
                                           p0=[m1_guess, m2_guess, m3_guess])
        self.m1, self.m2, self.m3 = params
        
    def malus_curve(self, x, m1, m2, m3):
        return m1 - m2 * (np.cos(np.deg2rad(x - m3))) ** 2

    def get_min_angle(self):
        """
        Find the minimum transmission angle based on fit parameters
        """
        return self.m3

    def plot_fit(self, ax):
        """
        Plot malus fit to intensities, with both data and fitted curve
        """
        fit_angles = np.linspace(np.min(self.angles), 
                                 np.max(self.angles), 
                                 100)
        fitted_intensities = self.malus_curve(fit_angles, 
                                              self.m1, 
                                              self.m2, 
                                              self.m3)
        ax.plot(fit_angles, fitted_intensities, color = "red")
        ax.scatter(self.angles, self.intensities)
        ax.set_title(f"{self.observation.mag_field_volt} V," 
                     f"{self.observation.filter},"
                     f"{self.observation.sample}")