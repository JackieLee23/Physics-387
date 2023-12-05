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
                 refr_ang = None, 
                 refl_ang = None, 
                 trans_int = None,
                 refl_int = None,
                 p_polarized = None):
        
        self.sample = sample
        self.interface = interface
        self.inc_ang = inc_ang
        self.refr_ang = refr_ang
        self.refl_ang = refl_ang
        self.trans_int = trans_int
        self.refl_int = refl_int
        self.p_polarized = p_polarized

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

    def fresnel(self, inc_ang, e, p, find_transmitted = True):
        """
        Angles in degrees
        p is ior ratio between incident and transmitting medium
        """
        inc_ang = np.deg2rad(inc_ang)
        trans_ang = np.arcsin(np.sin(inc_ang) / p)
        m = np.cos(trans_ang) / np.cos(inc_ang)

        if self.p_polarized:
            if find_transmitted:
                return e * p * m * (2 / (m + p)) ** 2
            else:
                return e * ((m - p) / (m + p)) ** 2
        else:
            if find_transmitted:
                return e * p * m * (2 / (1 + p * m))
            else:
                return e * ((1 - p * m) / (1 + p * m)) ** 2
            
    def fit_fresnel_reflected(self):
        """
        Fit reflection fresnel curve
        """
        
        self.valid_refl_inds = np.isfinite(self.refl_int)
        params, covariance = curve_fit(lambda x, e, p: self.fresnel(x, 
                                                                   e, 
                                                                   p, 
                                                                   find_transmitted = False), 
                                       self.inc_ang[self.valid_refl_inds], 
                                       self.refl_int[self.valid_refl_inds],
                                       p0 = [np.max(self.refl_int[self.valid_refl_inds]), 
                                             1])
        self.fresnel_refl_e, self.fresnel_refl_p = params

    def fit_fresnel_transmitted(self):
        self.valid_trans_inds = np.isfinite(self.trans_int)
        params, covariance = curve_fit(lambda x, e, p: self.fresnel(x, 
                                                                   e, 
                                                                   p, 
                                                                   find_transmitted = True), 
                                       self.inc_ang[self.valid_trans_inds], 
                                       self.trans_int[self.valid_trans_inds],
                                       p0 = [np.max(self.trans_int[self.valid_trans_inds]), 
                                             1])
        self.fresnel_trans_e, self.fresnel_trans_p = params

    def plot_fresnel_fit(self, ax):
        self.fit_fresnel_reflected()
        self.fit_fresnel_transmitted()
        fit_x = np.linspace(0,
                            90,
                            1000)
        ax.scatter(self.inc_ang, 
                   self.refl_int, 
                   label='Measured reflection intensities')
        ax.scatter(self.inc_ang,
                   self.trans_int,
                   label = 'Measured transmission intensity')
        ax.plot(fit_x, 
                self.fresnel(fit_x, self.fresnel_refl_e, self.fresnel_refl_p, False), 
                label='Fitted Curve', 
                color='red')
        ax.plot(fit_x, 
                self.fresnel(fit_x, self.fresnel_trans_e, self.fresnel_trans_p, True), 
                label='Fitted Curve', 
                color='red')
        
        
        ax.set_xlabel('Incident angle')
        ax.set_ylabel('Intensity angle')
        ax.set_title(f'{self.sample}, {self.interface}')



    





def invert_n(n, n_err):
    """
    Given an index of refraction ratio n, get the index of refraction ratio going in the opposite direction
    eg, given air to glass, get ior from glass to air
    """
    return 1/n, n_err * 1/(n**2)


    