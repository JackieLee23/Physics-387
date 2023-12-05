import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class AngleData:
    def __init__(self, 
                 sample, 
                 interface, 
                 inc_ang, 
                 refr_ang, 
                 refl_ang, 
                 trans_int = None,
                 refl_int = None):
        self.sample = sample
        self.interace = interface
        self.inc_ang = inc_ang
        self.refr_ang = refr_ang
        self.refl_ang = refl_ang
        self.trans_int = trans_int
        self.refl_int = refl_int