import numpy as np
import scipy as sp
import pickle

class EyeTriggers():
    def __init__(self):
        self.timestamp = None
        self.event_id  = None

class EyeHolder:
    def __init__(self):
        # self.fsamp = None
        self.info      = dict()
        self.triggers  = EyeTriggers()
        self.binocular = None
        self.eyes_recorded = []

class Blinks:
    def __init__(self, blinkarray):
        self.nblinks = blinkarray.shape[0]
        self.blinkstart = blinkarray[:,0]
        self.blinkend   = blinkarray[:,1]
        self.blinkdur   = blinkarray[:,2]