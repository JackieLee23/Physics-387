import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

ANGLE_ERROR = 0.5
ANGLE_ERROR_RAD = ANGLE_ERROR * np.pi/180

class AngleData:
    def __init__(self, 
                 sample, 
                 interface, 
                 inc_ang, 
                 refr_ang, 
                 refl_ang = None, 
                 trans_int = None,
                 refl_int = None):
        
        self.sample = sample
        self.interface = interface
        self.inc_ang = inc_ang
        self.refr_ang = refr_ang
        self.refl_ang = refl_ang
        self.trans_int = trans_int
        self.refl_int = refl_int

    def line(self, x, a, b):
        return a * x + b
    
    
    def fit_snell(self):
        valid_indices = np.isfinite(self.refr_ang) ##Have nan values sometimes, so make sure not to use those!
        snell_inc = np.deg2rad(self.inc_ang[valid_indices])
        snell_refr = np.deg2rad(self.refr_ang[valid_indices])

        self.inc_sines = np.sin(snell_inc)
        self.refr_sines = np.sin(snell_refr)

        ##Calculate uncertainties in sine of angles w propogation of errors:
        self.inc_sine_errs = ANGLE_ERROR_RAD * np.cos(snell_inc)
        self.refr_sine_errs = ANGLE_ERROR_RAD * np.cos(snell_refr)
        
        params, covariance = curve_fit(self.line, 
                                       self.inc_sines, 
                                       self.refr_sines,
                                       sigma = self.refr_sine_errs,
                                       absolute_sigma = True)
        self.snell_a, self.snell_b = params
        self.snell_a_err, self.snell_b_err = np.sqrt(np.diag(covariance))

    def plot_snell_fit(self, ax):
        self.fit_snell()
        fit_x = np.linspace(np.min(self.inc_sines),
                            np.max(self.inc_sines),
                            100)
        ax.errorbar(self.inc_sines, 
                    self.refr_sines, 
                    yerr=self.refr_sine_errs, 
                    xerr=self.inc_sine_errs, 
                    fmt='o', 
                    label='Data with Errors')
        ax.plot(fit_x, 
                self.line(fit_x, self.snell_a, self.snell_b), 
                label='Fitted Curve', 
                color='red')
        ax.set_xlabel('Sine of Incident angle')
        ax.set_ylabel('Sine of Refracted angle')
        ax.set_title(f'{self.sample}, {self.interface}')
        ax.legend()

def invert_n(n, n_err):
    """
    Given an index of refraction ratio n, get the index of refraction ratio going in the opposite direction
    eg, given air to glass, get ior from glass to air
    """
    return 1/n, n_err * 1/(n**2)


    