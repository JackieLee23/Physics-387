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
    def __init__(self, measurements):
        self.measurements = measurements

        ranges = findRanges(self.measurements)
        self.valueRanges = ranges.valueRanges
        self.keyVals = self.sampleFilterKeyList(self.valueRanges)
        self.BvsIntensityZeroField = self.BvsIntensityZeroField(self.measurements, self.keyVals)


    
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
    
    def BvsIntensityZeroField(self, measurements, keyVals): # a dictionary: {key = "{sample}_{filter}", value =  list of 3 numpy arrays: magfieldcurrents, intensities, associated zeroAngleFields}
        sample = "Sample (cm)" #Presumed headings of data in df
        filter = "Filter"
        zeroFieldAngle = "Zero Field Angle"
        intensity = "Intensity"
        magFieldCurrent = "Magnetic Field Current (Amps)"


        BIDict = {}

        for key, keyVal in keyVals.items():
    
            df = self.SFIsolate(measurements, keyVal) #Isolate the key's associated filter/sample pair in a df

            magFieldCurrents = df[magFieldCurrent].to_numpy() #get the magFieldCurrents of filter/sample pair
            intensities = df[intensity].to_numpy() #get the intensities of filter/sample pair
            zeroFieldAngles = df[zeroFieldAngle].to_numpy() #get the zero field angles of filter/sample pair

            BIDict[key] = [magFieldCurrents, intensities, zeroFieldAngles]

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


# data_file = 'data/faraday_data.xlsx'
# full_df = pd.read_excel(data_file)

# zeroFieldMalus(full_df)