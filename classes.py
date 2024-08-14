import numpy as np
import scipy as sp
import pickle
from copy import deepcopy

class EyeTriggers():
    def __init__(self):
        self.timestamp = None
        self.event_id  = None

class EyeHolder():
    def __init__(self):
        # self.fsamp = None
        self.info      = dict()
        self.triggers  = EyeTriggers()
        self.binocular = None
        self.eyes_recorded = []

    def drop_eye(self, eye_to_drop):
        '''
        this function drops one eye from the data structure, and amends the structure accordingly. From this point on, code will perceive it to be monocular and look for appropriate attributes
        '''
        eye = eye_to_drop.lower() #force lower case to identify the right attributes
        not_dropped = ['right' if eye == 'left' else 'left'][0]
        
        if self.binocular == False:
            raise TypeError('data are already monocular!')
        else:
            attrs_to_del = [x for x in self.__dict__.keys() if f'_{eye[0]}' in x]
            for iattr in attrs_to_del:
                delattr(self, iattr)
            attrs_to_rename = [x for x in self.__dict__.keys() if x[-2:] == f'_{not_dropped[0]}']
            for attr in attrs_to_rename:
                #rename attribute by creating a new one with the same values, then deleting the old one
                setattr(self, attr[:-2], getattr(self, attr))
                delattr(self, attr)
            
        setattr(self, 'binocular', False)
        setattr(self, 'eyes_recorded', [not_dropped])

    def average_channels(self, channels, new_name, func = 'nanmean'):
        '''
        function to average across multiple channels, and create a new channel in place. this is particularly useful if you want to average across e.g. both eyes, and create an average pupil channel
        note:
            can specify the function to apply, if you wish. it defaults to nanmean in case of missing data and to allow you to average where a channel is missing for a dataset
        '''
        #first just check that the channels have the same size or not
        sizes = [getattr(self, ichan).size for ichan in channels]
        if np.unique(sizes).size != 1:
            raise ValueError(f'channels to be averaged across have different trial durations')
        else:
            chans   = [getattr(self, ichan) for ichan in channels]
            chans   = np.vstack(chans)
            newchan = getattr(np, func)(chans, axis=0) #apply function across channels

            setattr(self, new_name, newchan)
            for ichan in channels:
                delattr(self, ichan)
    

    def rename_channel(self, old_name, new_name):
        '''
        rename an attribute (specifically intended for channels) -- mostly intended to help aligning channel names across datasets for later analysis
        '''
        if not hasattr(self, old_name):
            raise AttributeError(f'Attribute {old_name} could not be found')
        else:
            setattr(self, new_name, deepcopy(getattr(self, old_name))) #create new attribute with the same data
            delattr(self, old_name) #delete the old attribute name

class Blinks():
    def __init__(self, blinkarray):
        self.nblinks = blinkarray.shape[0]
        self.blinkstart = blinkarray[:,0]
        self.blinkend   = blinkarray[:,1]
        self.blinkdur   = blinkarray[:,2]