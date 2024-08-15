import numpy as np
import scipy as sp
import pandas as pd
import pickle

class epochedEyes():
    def __init__(self, data, srate, events, times, channels):
        self.data     = data
        self.srate    = srate
        self.event_id = events
        self.times    = times
        self.tmin     = times.min()
        self.metadata = None
        self.channels = channels
        self.blocks   = None

    def apply_baseline(self, baseline):
        bmin, bmax = baseline[0], baseline[1]
        [ntrls, nchannels, ntimes] = self.data.shape
        #get baseline value per trial
        blines = np.logical_and(np.greater_equal(self.times, bmin), np.less_equal(self.times, bmax))
        blinedata = self.data[:,:, blines].mean(axis=-1) #last axis is time, average across time in this window
        for itrl in range(ntrls):
            for ichan in range(nchannels):
                self.data[itrl, ichan] -= blinedata[itrl, ichan]
        return self

def concatenate_epochs(epoch_list):
    '''
    epochs - list of epoch structures to concatenate
    '''
    if len(epoch_list) > 2:
        raise ValueError('function can only handle concatenating two epoch instances at a time')
    else:
        for i, epochs in enumerate(epoch_list):
            if not isinstance(epochs, epochedEyes):
                raise TypeError(f'epoch_list[{i}] must be an instance of epochedEyes, got {type(epochs)}')
        
        e1, e2 = epoch_list[0], epoch_list[1]
        #need to check that certain attributes match between files
        attrs_to_check = ['srate', 'times', 'tmin', 'channels']
        badattrs = []
        for attr in attrs_to_check:
            same = None
            a1, a2 = getattr(e1, attr), getattr(e2, attr)
            if isinstance(a1, (np.ndarray, list)) or isinstance(a2, (np.ndarray, list)):
                same = np.array_equal(a1, a2)
            else:
                same = a1==a2
            if not same:
                badattrs.appen(attr)
        if len(badattrs) >0 :
            for attr in badattrs:
                raise Warning(f'mismatched values in the {attr} attribute between epoch instances')
            raise ValueError('Attribute values do not match between epoch instances, please check')
        
        #if you get this far, things seem ok to proceed
        d = np.vstack([e1.data, e2.data]) #vertically stack data so its [ntrials x nchannels x ntimes]
        t = np.hstack([e1.event_id, e2.event_id]) #horizontally concatenate triggers that were presented on each trial
        m = None

        e1m = isinstance(e1.metadata, pd.DataFrame) #evals as True if metadata is a dataframe
        e2m = isinstance(e2.metadata, pd.DataFrame) #evals as True if metadata is a dataframe
        if np.equal([e1m, e2m], True).sum() > 0: #check if there is any metadata available
            if e1m and not e2m:
                m = e1.metadata
            elif not e1m and e2m:
                m = e2.metadata
            elif e1m and e2m:
                m = pd.concat([e1.metadata, e2.metadata]) #concatenate
        
        #some attributes can be taken from either data set as they are checked to be equal, so it doesn't matter where its from
        newepochs = epochedEyes(data = d,   #add in concatenated data
                                srate = e1.srate, #take sample rate from either, as they are the t
                                events = t, #assign the triggers
                                times = e1.times,
                                channels = e1.channels)
        if not isinstance(m, type(None)):
            setattr(newepochs, 'metadata', m) #assign metadata
        return newepochs
