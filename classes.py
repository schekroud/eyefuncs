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


class Blinks():
    def __init__(self, blinkarray):
        self.nblinks = blinkarray.shape[0]
        self.blinkstart = blinkarray[:,0]
        self.blinkend   = blinkarray[:,1]
        self.blinkdur   = blinkarray[:,2]