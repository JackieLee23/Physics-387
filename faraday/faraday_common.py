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
        
class minEyeTransmission:
    def __init__(self, measurements):
        self.measurements = measurements

        valRanges = findRanges(measurements)
        self.ranges = valRanges.valueRanges
        self.keyVals = self.sampleFilterKeyList(self.ranges)
        self.transmissionAnglesVersusBField = self.angleVsB(self.measurements, self.keyVals)
        

    def sampleFilterKeyList(self, ranges): #this will create a dictionary of keys in our returned dictionary that will have the structure of: "{Sample}_{Filter}" for each key and store as a tuple (Sample, filter)
        
        sample = "Sample (cm)" #Presumed names of sample and filter data in directory
        filter = "Filter"

        
        sampleFilterKeys = {} #Dict of keys

        samples = ranges[sample] #Grabbing the unique sample and filter values
        filters = ranges[filter]

        for s in samples:
            for f in filters:
                sampleFilterKeys[f"{s}_{f}"] = (s,f)
        
        return sampleFilterKeys


# popt, pcov = curve_fit(f, x, y) # your data x, y to fit
    
    def angleVsB(self, measurements, keyVals): #Creates a dictionary with values of transmission angle versus B field for all sample/filter combinations
                                                #Returns dictionary: {key = "{sample}+{filter}", val = [angleVals, BFieldVals]} for all sample/filter permutations
        sample = "Sample (cm)" #Presumed headings of data in df
        filter = "Filter"
        magField = "Magnetic Field Current (Amps)"
        angle = "Min transmission angle"

        angleVsB = {} #Angles versus magnetic field for all values

        for key, values in keyVals.items(): #Iterating over each sample/filter permutation
            magneticFieldVals =  measurements[(measurements[sample] == values[0]) & (measurements[filter] == values[1])][magField].to_numpy() #Collects B field values for permutation
            angleVals = measurements[(measurements[sample] == values[0]) & (measurements[filter] == values[1])][angle].to_numpy() #Collects transmission angle values for permutation

            angleVsB[key] = [angleVals, magneticFieldVals]
        
        return angleVsB
    


class zeroFieldMalus:
    def __init__(self, measurements, mList, b, m):
        self.b = b
        self.m = m
        self.measurements = measurements
        self.mList = mList

        ranges = findRanges(self.measurements)
        self.valueRanges = ranges.valueRanges
        self.keyVals = self.sampleFilterKeyList(self.valueRanges)
        self.BvsIntensityZeroField = self.BvsIntensityZeroField(self.measurements, self.keyVals, self.mList, self.m, self.b)
        


    
    def sampleFilterKeyList(self, ranges): #this will create a dictionary of keys in our returned dictionary that will have the structure of: "{Sample}_{Filter}" for each key and store as a tuple (Sample, filter)
        
        sample = "Sample (cm)" #Presumed names of sample and filter data in directory
        filter = "Filter"

        
        sampleFilterKeys = {} #Dict of keys

        samples = ranges[sample] #Grabbing the unique sample and filter values
        filters = ranges[filter]

        for s in samples:
            for f in filters:
                sampleFilterKeys[f"{s}_{f}"] = (s,f)
        
        return sampleFilterKeys 
    
    def SFIsolate(self, measurements, keyVal): #Gives me a dataframe of measurements of a particular {Sample}_{Filter} combination
        sample = "Sample (cm)" #Presumed headings of data in df
        filter = "Filter"

        df = measurements[(measurements[sample] == keyVal[0]) & (measurements[filter] == keyVal[1])]
        
        return df
    def rotationAngle(self, V, m1, m2, m3): #Formula for a single rotation angle given m_i fitting parameters and a magnetic field dependent voltage. Note that V here is a scalar, rotation angle handles arrays
        
        phi = np.rad2deg(np.arccos(np.sqrt((m1-V)/m2)))+m3
        return phi
    
    def rotationAngles(self, Vs, m1, m2, m3):
        phis = []
        
        for V in Vs:
            phis.append(self.rotationAngle(V, m1, m2, m3))
        
        return np.array(phis)

    def magneticFieldVals(self, current, m, b):
        return np.add(np.multiply(m,current), b)

    def BvsIntensityZeroField(self, measurements, keyVals, mList, m, b): # a dictionary: {key = "{sample}_{filter}", value =  list of 3 numpy arrays: magfieldcurrents, intensities, associated zeroAngleFields}
        sample = "Sample (cm)" #Presumed headings of data in df
        filter = "Filter"
        zeroFieldAngle = "Zero Field Angle"
        intensity = "Intensity"
        magFieldCurrent = "Magnetic Field Current (Amps)"


        BIDict = {}

        for key, keyVal in keyVals.items():
            m1 = mList[key]["m1"]
            m2 = mList[key]["m2"]
            m3 = mList[key]["m3"]

            df = self.SFIsolate(measurements, keyVal) #Isolate the key's associated filter/sample pair in a df

            magFieldCurrents = df[magFieldCurrent].to_numpy() #get the magFieldCurrents of filter/sample pair
            intensities = df[intensity].to_numpy() #get the intensities of filter/sample pair
            zeroFieldAngles = df[zeroFieldAngle].to_numpy() #get the zero field angles of filter/sample pair
            phis = self.rotationAngles(intensities, m1, m2, m3)
            magneticFieldVals = self.magneticFieldVals(magFieldCurrents, m, b)


            BIDict[key] = {"current": magFieldCurrents, 
                           "intensities": intensities, 
                           "zero Field angles": zeroFieldAngles,
                           "rotation angles" : phis,
                           "magnetic field values" : magneticFieldVals
                           }

        return BIDict
    
    


            



class findRanges: #Finds the different, unique values in the measurements dataframe
    def __init__(self, measurements):
        self.measurements = measurements
        self.headings = measurements.columns #The identifiers
        self.valueRanges = self.valueRanges(self.measurements, self.headings) # a dictionary: {key = heading name from self.headings, value = different unique values for heading}
    
    def valueRanges(self, measurements, headings):
        valueRanges = {}
        for heading in headings:
            repetitions = measurements[heading].duplicated(keep = "first")
            valueRanges[heading] = measurements[~repetitions][heading].to_numpy()
        return valueRanges
"""
Testing data below, make sure to comment out when using Jupyter
"""


data_file = 'data/faraday_data.xlsx'
full_df = pd.read_excel(data_file)
mList = {'1.036_Red': {'m1': 0.07320082950131986,
  'm2': 0.07242662219366228,
  'm3': 335.274925890484},
 '0.956_Red': {'m1': 0.07600340873265846,
  'm2': 0.07524317421585733,
  'm3': 338.28784039026385},
 '1.272_Red': {'m1': 0.058755378942364996,
  'm2': 0.05742480689139942,
  'm3': 335.95791532842117},
 '1.272_Blue': {'m1': 0.11509943462188175,
  'm2': 0.11027621309592965,
  'm3': 334.2244603785799},
 '0.956_Blue': {'m1': 0.13607536773223358,
  'm2': 0.12719212466437843,
  'm3': 336.57688964189674},
 '1.036_Blue': {'m1': 0.10586229718833813,
  'm2': 0.10266727840879811,
  'm3': 336.1480069830858},
 '1.036_Yellow': {'m1': 0.07614245258643797,
  'm2': 0.0742616330734973,
  'm3': 335.6961290674026},
 '0.956_Yellow': {'m1': 0.09596294707418054,
  'm2': 0.09173222470975656,
  'm3': 336.1317016374635},
 '1.272_Yellow': {'m1': 0.0876601595037936,
  'm2': 0.08466557881776582,
  'm3': 336.54722589233563}}
m = 190.86
b = -0.0152

zeroFieldMalus(full_df, mList, b, m)