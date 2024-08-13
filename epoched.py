import numpy as np
import scipy as sp
import pickle

class epochedEyes():
    def __init__(self, data, srate, events, times):
        self.data     = data
        self.srate    = srate
        self.event_id = events
        self.times    = times
        self.tmin     = times.min()
        self.metadata = None
        
    
    def apply_baseline(self, baseline):
        bmin, bmax = baseline[0], baseline[1]
        #get baseline value per trial
        blines = np.logical_and(np.greater_equal(self.times, bmin), np.less_equal(self.times, bmax))
        blinedata = self.data[:,blines].mean(axis=1) #average across time in this window
        for itrl in range(self.data.shape[0]):
            self.data[itrl] -= blinedata[itrl]
        
        return self