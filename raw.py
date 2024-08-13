import numpy as np
import scipy as sp
from copy import deepcopy
import pickle
from .utils import _calculate_blink_periods

class rawEyes:
    def __init__(self, nblocks, srate):
        self.nblocks = nblocks
        self.data    = list()
        self.srate   = srate
        self.fsamp   = None
        self.binocular = None
    
    def nan_missingdata(self):
        for iblock in range(self.nblocks): #loop over blocks
            tmpdata = self.data[iblock]
            
            if tmpdata.binocular: 
                for ieye in tmpdata.eyes_recorded:
                    missinds = np.where(getattr(tmpdata, 'pupil_'+ieye[0]) == 0)[0] #missing data is assigned to 0 for pupil trace
                    #replace missing values with nan
                    traces = [f'{x}_{ieye[0]}' for x in ['pupil', 'xpos', 'ypos']] #get attribute labels to loop over
                    for trace in traces:
                        getattr(tmpdata, trace)[missinds] = np.nan
            else:
                missinds = np.where(tmpdata.pupil == 0) #missing data is assigned to 0 for pupil trace
                for trace in ['pupil', 'xpos', 'ypos']: #nan everything in these channels where the pupil is zero
                    getattr(tmpdata, trace)[missinds] = np.nan
                
            self.data[iblock] = tmpdata
    
    def identify_blinks(self, buffer = 0.150, add_nanchannel = True):
        #set up some parameters for the algorithm
        blinkspd        = 2.5                     #speed above which data is remove around nan periods -- threshold
        maxvelthresh    = 30
        maxpupilsize    = 20000
        cleanms         = buffer * self.srate     #ms padding around the blink edges for removal
        
        for iblock in range(self.nblocks): #loop over blocks in the data
            blockdata = deepcopy(self.data[iblock])
            binocular = blockdata.binocular
            if binocular:
                blockdata = _find_blinks_binocular(blockdata, self.srate, buffer, add_nanchannel, blinkspd, maxvelthresh, maxpupilsize, cleanms)
            elif not binocular:
                blockdata = _find_blinks_monocular(blockdata, self.srate, buffer, add_nanchannel, blinkspd, maxvelthresh, maxpupilsize, cleanms)
            
            self.data[iblock] = blockdata #assign back into the data structure
    
    def interpolate_blinks(self):
        for iblock in range(self.nblocks):
            if self.data[iblock].binocular:
                self.data[iblock] = _interpolate_blinks_binocular(self.data[iblock])
            else:
                self.data[iblock] = _interpolate_blinks_monocular(self.data[iblock])

    
    def smooth_pupil(self, sigma = 50):
        '''
        smooth the clean pupil trace with a gaussian with standard deviation sigma
        '''
        for iblock in range(self.nblocks):
            blockdata = deepcopy(self.data[iblock])
            if blockdata.binocular:
                for eye in blockdata.eyes_recorded:
                    ieye = eye[0]
                    att = f'pupil_{ieye}_clean'
                    if not hasattr(blockdata, att):
                        raise AttributeError(f'Attribute not found: could not find {att}')
                    else:
                        setattr(blockdata, f'pupil_{ieye}_clean',
                                sp.ndimage.gaussian_filter1d(getattr(blockdata, f'pupil_{ieye}_clean'), sigma=sigma) #smooth signal
                                )
            elif not blockdata.binocular:
                if not hasattr(blockdata, 'pupil_clean'):
                    raise AttributeError('Attribute not found: could not find "pupil_clean"')
                else:
                    setattr(blockdata, 'pupil_clean',
                            sp.ndimage.gaussian_filter1d(getattr(blockdata, 'pupil_clean'), sigma=sigma) #smooth signal
                    )
            self.data[iblock] = blockdata

                
    def save(self, fname):
        with open(fname, 'wb') as handle:
            pickle.dump(self, handle)
            
    def cubicfit(self):
        #define cubic function to be fit to the data
        def cubfit(x, a, b, c, d):
            return a*np.power(x,3) + b*np.power(x, 2) + c*np.power(x,1) + d
        
        for iblock in range(self.nblocks):
            tmpdata = self.data[iblock]
            if not tmpdata.binocular:
                fitparams = sp.optimize.curve_fit(cubfit, tmpdata.time, tmpdata.pupil_clean)[0]
                modelled  = fitparams[0]*np.power(tmpdata.time, 3) + fitparams[1]*np.power(tmpdata.time, 2) + fitparams[2]*np.power(tmpdata.time, 1) + fitparams[3]
                diff = tmpdata.pupil_clean - modelled #subtract this cubic fit
                #assign modelled data and the corrected data back into the data structure
                
                self.data[iblock].modelled        = modelled
                self.data[iblock].pupil_corrected = diff
            elif tmpdata.binocular:
                for eye in tmpdata.eyes_recorded:
                    ieye = eye[0] #get suffix used to get the right data
                    ip = getattr(tmpdata, f'pupil_{ieye}_clean').copy()
                    fitparams = sp.optimize.curve_fit(cubfit, tmpdata.time, ip)[0]
                    modelled  = fitparams[0]*np.power(tmpdata.time, 3) + fitparams[1]*np.power(tmpdata.time, 2) + fitparams[2]*np.power(tmpdata.time, 1) + fitparams[3]
                    diff = ip - modelled #subtract this cubic fit
                    setattr(self.data[iblock], f'pupil_{ieye}_corrected', diff)
                    setattr(self.data[iblock], f'modelled_{ieye}', modelled)

    
    def transform_channel(self, channel, method = 'percent'):
        for iblock in range(self.nblocks): #loop over blocks in the data
            tmpdata = self.data[iblock].__getattribute__(channel).copy()
            transformed = np.zeros_like(tmpdata)
            if method == 'zscore':
                transformed = sp.stats.zscore(tmpdata)
            elif method == 'percent':
                mean = tmpdata.mean()
                transformed = np.subtract(tmpdata, mean)
                transformed = np.multiply(np.divide(transformed, mean), 100)
            self.data[iblock].__setattr__('pupil_transformed', transformed) #save the transformed data back into the data object
            #self.data[iblock].pupil_transformed = transformed 
            
def _find_blinks_binocular(data, srate, buffer, add_nanchannel, blinkspd, maxvelthresh, maxpupilsize, cleanms):
    '''
    data - a single block of recorded data (class: EyeHolder)
    '''
    idata = data
    eyesrec = data.eyes_recorded
    for eye in eyesrec:
        ieye = eye[0] #the string for getting the data
        pupil = getattr(data, 'pupil_'+ieye) #get the pupil trace for this eye
        iblinks, nantrace = _calculate_blink_periods(pupil, srate, blinkspd, maxvelthresh, maxpupilsize, cleanms)
        setattr(idata, 'blinks_'+ieye, iblinks)
        if add_nanchannel:
            setattr(idata, f'pupil_{ieye}_nan', nantrace) #assign nan channel for this eye
    return idata

def _find_blinks_monocular(data, srate, buffer, add_nanchannel, blinkspd, maxvelthresh, maxpupilsize, cleanms):
    '''
    data - a single block of recorded data (class: EyeHolder)
    '''
    idata = deepcopy(data)
    pupil = data.pupil
    iblinks, nantrace = _calculate_blink_periods(pupil, srate, blinkspd, maxvelthresh, maxpupilsize, cleanms)
    setattr(idata, 'blinks', iblinks)
    if add_nanchannel:
        setattr(idata, 'pupil_nan', nantrace) #assign nan channel for this eye
    
    return idata

def _interpolate_blinks_monocular(data):
    '''
    data - a single block of recorded data (class: EyeHolder)
    '''
    idata    = deepcopy(data)
    pupil    = idata.pupil.copy()
    nanpupil = idata.pupil_nan.copy()
    times    = idata.time.copy()
    
    mask = np.zeros_like(times, dtype=bool)
    mask |= np.isnan(nanpupil)
    
    interpolated = np.interp(
        times[mask],
        times[~mask],
        pupil[~mask]
        )
    
    cleanpupil = nanpupil.copy()
    cleanpupil[mask] = interpolated
    setattr(idata, 'pupil_clean', cleanpupil)
    return idata

def _interpolate_blinks_binocular(data):
    '''
    data - a single block of recorded data (class: EyeHolder)
    '''
    idata = deepcopy(data)
    for eye in idata.eyes_recorded:
        ieye = eye[0] #get suffix label
        pupil    = getattr(idata, f'pupil_{ieye}').copy()
        nanpupil = getattr(idata, f'pupil_{ieye}_nan').copy()
        times    = getattr(idata, 'time').copy()

        mask = np.zeros_like(times, dtype=bool)
        mask |= np.isnan(nanpupil)

        interpolated = np.interp(
            times[mask],
            times[~mask],
            pupil[~mask]
        )
        cleanpupil = nanpupil.copy()
        cleanpupil[mask] = interpolated
        setattr(idata, f'pupil_{ieye}_clean', cleanpupil)
    return idata
