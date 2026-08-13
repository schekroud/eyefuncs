import numpy as np
import scipy as sp
import scipy.ndimage as ndi
from copy import deepcopy
from .utils import smooth
from .classes import Blinks


class rawEyes():
    def __init__(self, nblocks, srate):
        self.nblocks = nblocks
        self.data    = list()
        self.srate   = srate
        self.fsamp   = None
        self.binocular = None
    
    def nan_missingdata(self, align_eyes=False):
        for iblock in range(self.nblocks): #loop over blocks
            tmpdata = self.data[iblock]
            
            if tmpdata.binocular: 
                missing = np.vstack([
                    np.equal(getattr(tmpdata, 'pupil_l'),0),
                    np.equal(getattr(tmpdata, 'pupil_r'),0)
                ])
                aligned = np.greater(missing.sum(0), 0) #if sum if data is missing across the eyes. if 0, both present. if 1, 1 eye has missing data. 2 = both eyes missing that timepoint
                if not align_eyes:
                    for ieye in tmpdata.eyes_recorded:
                        ie = ieye[0]
                        if ie=='l':
                            imissing=missing[0]
                        else:
                            imissing=missing[1]
                        missinds = np.where(imissing)[0]
                        traces = [f'{x}_{ie}' for x in ['pupil', 'xpos', 'ypos']]
                        for trace in traces:
                            getattr(tmpdata,trace)[missinds]=np.nan
                else: #if aligning missing data from the two eyes
                    missinds =np.where(aligned)[0]
                    traces = [f'{x}_{y[0]}' for x in ['pupil', 'xpos', 'ypos'] for y in tmpdata.eyes_recorded]
                    for trace in traces:
                        getattr(tmpdata, trace)[missinds] = np.nan
            else:
                missinds = np.where(tmpdata.pupil == 0) #missing data is assigned to 0 for pupil trace
                for trace in ['pupil', 'xpos', 'ypos']: #nan everything in these channels where the pupil is zero
                    getattr(tmpdata, trace)[missinds] = np.nan
                
            self.data[iblock] = tmpdata
    
    def identify_blinks(self, buffer = 0.150, add_nanchannel = True, bridge_time = 0.02, align_eyes = True):
        """
        detect blink-related data missingness and store:
        - per-eye raw masks
        - shared binocular mask (if align_eyes = True)
        - blink structure (s)
        - pupil trace with nan'd data
        """
        
        buffer_samples = int(buffer * self.srate)
        bridge_samples = None if bridge_time is None else int(bridge_time * self.srate) #if there is a period of up to this duration between blinks, it bridges the gap to treat as one blink.    
        
        for iblock in range(self.nblocks):
            if self.data[iblock].binocular:
                nanmasks = _detect_blinks_binocular(self.data[iblock], buffer_samples, bridge_samples)
                if align_eyes:
                    aligned = np.logical_or(nanmasks[0], nanmasks[1])
                eyesrec = self.data[iblock].eyes_recorded
                ieyes = [x[0] for x in eyesrec] #should just output 'l' or 'r'
                for (i, ie) in enumerate(ieyes):
                    if align_eyes:
                        setattr(self.data[iblock], f'nanmask_{ie}', aligned)
                    else:
                        setattr(self.data[iblock], f'nanmask_{ie}', nanmasks[i])
                
                    #create blink structure
                    iblinks = _create_blink_structure(getattr(self.data[iblock], f'nanmask_{ie}'), self.srate)
                    setattr(self.data[iblock], f'blinks_{ie}', iblinks)
                    
                    if add_nanchannel:
                        ipupil = getattr(self.data[iblock], f'pupil_{ie}')
                        imask  = getattr(self.data[iblock], f'nanmask_{ie}')
                        nantrace = ipupil.copy()
                        nantrace[imask] = np.nan
                        setattr(self.data[iblock], f'pupil_nan_{ie}', nantrace)
            else:
                nanmask = _detect_blinks_monocular(self.data[iblock], buffer_samples, bridge_samples)
                setattr(self.data[iblock], 'nanmask', nanmask)
                iblinks = _create_blink_structure(getattr(self.data[iblock], 'nanmask'), self.srate)
                setattr(self.data[iblock], 'blinks', iblinks)
                
                if add_nanchannel:
                    ipupil = getattr(self.data[iblock], 'pupil')
                    imask  = getattr(self.data[iblock], 'nanmask')
                    nantrace = ipupil.copy()
                    nantrace[imask] = np.nan
                    setattr(self.data[iblock], 'pupil_nan', nantrace)
            
            self.data[iblock].info['blinks_identified'] = True
                
    
    def interpolate_blinks(self):
        for iblock in range(self.nblocks):
            if self.data[iblock].binocular:
                self.data[iblock] = _interpolate_blinks_binocular(self.data[iblock])
            else:
                nsamps = self.data[iblock].trackertime.size
                if np.isnan(self.data[iblock].pupil).sum() == nsamps: #data missing for entire block
                    self.data[iblock].info['full_block_missing'] = True
                    setattr(self.data[iblock], 'pupil_clean', np.zeros(nsamps)*np.nan)
                else:
                    self.data[iblock].info['full_block_missing'] = False
                    self.data[iblock] = _interpolate_blinks_monocular(self.data[iblock])
            self.data[iblock].info['blinks_cleaned'] = True #log that this step has happened

    def drop_eye(self, eye_to_drop):
        '''
        this function drops one eye from the data structure, and amends the structure accordingly. From this point on, code will perceive it to be monocular and look for appropriate attributes
        
        eye_to_drop can either be a single string ('left', 'right) or a list of strings ['none', 'left', 'right']. 
        If a single string is passed, that eye is dropped for the entire object.
        If a list of strings is passed, the eye to drop can vary across blocks of the task. it iterates over blocks to drop (or not) specific eyes. recodes each relevant block as monocular
        '''
        nblocks = self.nblocks
        mapping = {'left':'right', 'right':'left', 'none':'none'} #if left is dropped, right is not dropped. vice versa. preserve nones (no removal)
        if not isinstance(eye_to_drop, list): #we are removing the same eye from all blocks in this case, so just make a list for each block
            eyes2rem = [eye_to_drop] * nblocks
        else:
            eyes2rem = eye_to_drop
        not_dropped = [mapping.get(x, 'none') for x in eyes2rem] #get the eye that wasnt dropped
        for iblock in range(nblocks):
            if self.data[iblock].binocular == False: #already monocular
                print(f'skipping block {iblock+1} as data are already monocular')
            else:
                if eyes2rem[iblock] != 'none':
                    tmpdata = deepcopy(self.data[iblock])
                    attrs_to_del = [x for x in tmpdata.__dict__.keys() if x.endswith(f'_{eyes2rem[iblock][0]}')]
                    for iattr in attrs_to_del:
                        delattr(tmpdata, iattr)
                    attrs_to_rename = [x for x in tmpdata.__dict__.keys() if x.endswith(f'_{not_dropped[iblock][0]}')]
                    for attr in attrs_to_rename:
                        #rename attribute by creating a new one with the same values, then deleting the old one
                        setattr(tmpdata, attr[:-2], getattr(tmpdata, attr))
                        delattr(tmpdata, attr)
                    setattr(tmpdata, 'binocular', False)
                    setattr(tmpdata, 'eyes_recorded', [not_dropped[iblock]])
                    self.data[iblock] = tmpdata
        #there is a case where the same eye is removed from all blocks, or one eye is removed from all blocks (so its effectively monocular). check this and assign monocularity if so
        binoccheck = np.sum([x.binocular for x in self.data])
        if binoccheck == 0: #no binocular blocks found
            self.binocular = False
        #if there are any binocular blocks it'll leave it as binocular
     
    def smooth_pupil(self, sigma = 50):
        '''
        smooth the clean pupil trace with a gaussian with standard deviation sigma
        '''
        for iblock in range(self.nblocks):
            blockdata = deepcopy(self.data[iblock])
            if blockdata.binocular:
                for eye in blockdata.eyes_recorded:
                    ieye = eye[0]
                    att = f'pupil_clean_{ieye}'
                    if not hasattr(blockdata, att):
                        raise AttributeError(f'Attribute not found: could not find {att}')
                    else:
                        setattr(blockdata, f'pupil_clean_{ieye}',
                                sp.ndimage.gaussian_filter1d(getattr(blockdata, f'pupil_clean_{ieye}'), sigma=sigma) #smooth signal
                                )
            elif not blockdata.binocular:
                if not hasattr(blockdata, 'pupil_clean'):
                    raise AttributeError('Attribute not found: could not find "pupil_clean"')
                else:
                    if not blockdata.info['full_block_missing']: #dont do anything if the full block is missing
                        setattr(blockdata, 'pupil_clean',
                                sp.ndimage.gaussian_filter1d(getattr(blockdata, 'pupil_clean'), sigma=sigma) #smooth signal
                        )
            self.data[iblock] = blockdata

    def cubicfit(self):
        #define cubic function to be fit to the data
        def cubfit(x, a, b, c, d):
            return a*np.power(x,3) + b*np.power(x, 2) + c*np.power(x,1) + d
        
        for iblock in range(self.nblocks):
            tmpdata = self.data[iblock]
            if not tmpdata.binocular and not tmpdata.info['full_block_missing']: #cant model when full block recorded is bad
                fitparams = sp.optimize.curve_fit(cubfit, tmpdata.time, tmpdata.pupil_clean)[0]
                modelled  = fitparams[0]*np.power(tmpdata.time, 3) + fitparams[1]*np.power(tmpdata.time, 2) + fitparams[2]*np.power(tmpdata.time, 1) + fitparams[3]
                diff = tmpdata.pupil_clean - modelled #subtract this cubic fit
                #assign modelled data and the corrected data back into the data structure
                
                self.data[iblock].modelled        = modelled
                self.data[iblock].pupil_corrected = diff
            elif tmpdata.binocular:
                for eye in tmpdata.eyes_recorded:
                    ieye = eye[0] #get suffix used to get the right data
                    ip = getattr(tmpdata, f'pupil_clean_{ieye}').copy()
                    fitparams = sp.optimize.curve_fit(cubfit, tmpdata.time, ip)[0]
                    modelled  = fitparams[0]*np.power(tmpdata.time, 3) + fitparams[1]*np.power(tmpdata.time, 2) + fitparams[2]*np.power(tmpdata.time, 1) + fitparams[3]
                    diff = ip - modelled #subtract this cubic fit
                    setattr(self.data[iblock], f'pupil_corrected_{ieye}', diff)
                    setattr(self.data[iblock], f'modelled_{ieye}', modelled)
            self.data[iblock].info['pupil_corrected'] = True #log that this step has happened

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
            self.data[iblock].info['pupil_transformed'] = True #log that this step has happened

def _detect_blinkmask_single_eye(pupil, buffer_samples, bridge_samples, blinkspd = 2.5, maxvelthresh = 30, maxpupilsize = 20000):
    '''
    
    '''
    signal = pupil.copy()
    vel    = np.diff(pupil) #derivative of pupil diameter
    speed  = np.abs(vel)    #absolute velocity
    smoothv   = smooth(vel, twin = 8, method = 'boxcar') #smooth with a 8ms boxcar to remove tremor in signal
    smoothspd = smooth(speed, twin = 8, method = 'boxcar') #smooth to remove some tremor
    #not sure if it quantitatively changes anything if you use a gaussian instead. the gauss filter makes it smoother though

    #missing data should have already been set to nan, find them:
    zerosamples = np.isnan(pupil) #check where data is missing.
    
    #create an array logging bad samples in the trace
    badsamples = np.zeros_like(pupil, dtype=bool)
    badsamples[1:] = np.logical_or(speed >= maxvelthresh, pupil[1:] > maxpupilsize)
    
    #expand the periods detected here with a buffer
    badsamples = ndi.binary_dilation(badsamples, structure = np.ones(int(2*buffer_samples + 1)))
    badsamps = (badsamples | zerosamples) #get whether its marked as a bad sample, OR marked as a previously zero sample ('blinks' to be interpolated)

    if bridge_samples != None and bridge_samples >1:
        badsamps = ndi.binary_closing(badsamps, structure = np.ones(bridge_samples)) #this will merge blinks that occur in rapid succession with a short (noisy) period of recorded data between
    # signal[badsamps==1] = np.nan #set these bad samples to nan
    return badsamps #return the boolean mask that marks bad data that should be set to nan

def _detect_blinks_binocular(data, buffer_samples, bridge_samples):
    '''
    takes blockdata
    '''
    eyesrec = data.eyes_recorded
    ieyes = [x[0] for x in eyesrec] #should just output 'l' or 'r'
    masks = [None] * len(ieyes)
    for (i, ie) in enumerate(ieyes):
        masks[i] = _detect_blinkmask_single_eye(getattr(data, f'pupil_{ie}'), buffer_samples, bridge_samples)
    masks = np.vstack(masks) #stack into an array
    return masks

def _detect_blinks_monocular(data, buffer_samples, bridge_samples):
    '''
    '''
    mask = _detect_blinkmask_single_eye(getattr(data, 'pupil'), buffer_samples, bridge_samples)
    return mask
    

def _create_blink_structure(mask, srate):
    
    '''
    '''
    
    changebads = np.zeros_like(mask, dtype = int)
    changebads[1:] = np.diff(mask.astype(int))
    #starts are always off by one sample - when changebads == 1, the data is now MISSING. we need the sample before for interpolation
    starts = np.squeeze(np.where(changebads==1)) -1
    ends = np.squeeze(np.where(changebads==-1))

    if starts.size != ends.size:
        print(f"There is a problem with your data and the start/end of blinks dont match.\n- There are {starts.size} blink starts and {ends.size} blink ends")
        if starts.size == ends.size - 1:
            print('The recording starts on a blink; fixing')
            starts = np.insert(starts, 0, 0, 0)
        if starts.size == ends.size + 1:
            print('The recording ends on a blink; fixing')
            ends = np.append(ends, len(mask))
    durations = np.divide(np.subtract(ends, starts), srate) #get duration of each saccade in seconds

    blinkarray = np.array([starts, ends, durations]).T
    blinks = Blinks(blinkarray)
    return blinks

def _find_blinks_binocular(data, srate, buffer, add_nanchannel, blinkspd, maxvelthresh, maxpupilsize, cleanms, bridge_samples):
    '''
    data - a single block of recorded data (class: EyeHolder)
    '''
    idata = data
    eyesrec = data.eyes_recorded
    for eye in eyesrec:
        ieye = eye[0] #the string for getting the data
        pupil = getattr(data, 'pupil_'+ieye) #get the pupil trace for this eye
        iblinks, nantrace = _calculate_blink_periods(pupil, srate, blinkspd, maxvelthresh, maxpupilsize, cleanms, bridge_samples)
        setattr(idata, 'blinks_'+ieye, iblinks)
        if add_nanchannel:
            setattr(idata, f'pupil_nan_{ieye}', nantrace) #assign nan channel for this eye
    return idata

def _find_blinks_monocular(data, srate, buffer, add_nanchannel, blinkspd, maxvelthresh, maxpupilsize, cleanms, bridge_samples):
    '''
    data - a single block of recorded data (class: EyeHolder)
    '''
    idata = deepcopy(data)
    pupil = data.pupil
    iblinks, nantrace = _calculate_blink_periods(pupil, srate, blinkspd, maxvelthresh, maxpupilsize, cleanms, bridge_samples)
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
        nanpupil = getattr(idata, f'pupil_nan_{ieye}').copy()
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
        setattr(idata, f'pupil_clean_{ieye}', cleanpupil)
    return idata

def _calculate_blink_periods(pupil, srate,  blinkspd, maxvelthresh, maxpupilsize, cleanms, bridge_samples):
    signal = pupil.copy()
    vel    = np.diff(pupil) #derivative of pupil diameter
    speed  = np.abs(vel)    #absolute velocity
    smoothv   = smooth(vel, twin = 8, method = 'boxcar') #smooth with a 8ms boxcar to remove tremor in signal
    smoothspd = smooth(speed, twin = 8, method = 'boxcar') #smooth to remove some tremor
    #not sure if it quantitatively changes anything if you use a gaussian instead. the gauss filter makes it smoother though
    
    #pupil size only ever reaches zero if missing data. so we'll log this as missing data anyways
    # zerosamples = np.zeros_like(pupil, dtype=bool)
    # zerosamples[pupil==0] = True

    #missing data should have already been set to nan, not zero:
    zerosamples = np.isnan(pupil) #check where data is missing.
    #if you work on the assumption that the eyelink accurately identifies blinks and set pupil to 0, we can smooth out this with a buffer period to capture the blink artefact
    
    
    #create an array logging bad samples in the trace
    badsamples = np.zeros_like(pupil, dtype=bool)
    badsamples[1:] = np.logical_or(speed >= maxvelthresh, pupil[1:] > maxpupilsize)
    
    #expand the periods detected here with a buffer
    badsamples = ndi.binary_dilation(badsamples, structure = np.ones(int(2*cleanms + 1)))
    badsamps = (badsamples | zerosamples) #get whether its marked as a bad sample, OR marked as a previously zero sample ('blinks' to be interpolated)

    if bridge_samples != None:
        badsamps = ndi.binary_closing(badsamps, structure = np.ones(bridge_samples)) #this will merge blinks that occur in rapid succession with a short (noisy) period of recorded data between
    signal[badsamps==1] = np.nan #set these bad samples to nan
    
    #we want to  create 'blink' structures, so we need info here
    changebads = np.zeros_like(pupil, dtype=int)
    changebads[1:] = np.diff(badsamps.astype(int)) #+1 = from not missing -> missing; -1 = missing -> not missing

    #starts are always off by one sample - when changebads == 1, the data is now MISSING. we need the sample before for interpolation
    starts = np.squeeze(np.where(changebads==1)) -1
    ends = np.squeeze(np.where(changebads==-1))

    if starts.size != ends.size:
        print(f"There is a problem with your data and the start/end of blinks dont match.\n- There are {starts.size} blink starts and {ends.size} blink ends")
        if starts.size == ends.size - 1:
            print('The recording starts on a blink; fixing')
            starts = np.insert(starts, 0, 0, 0)
        if starts.size == ends.size + 1:
            print('The recording ends on a blink; fixing')
            ends = np.append(ends, len(pupil))

    durations = np.divide(np.subtract(ends, starts), srate) #get duration of each saccade in seconds
    
    blinkarray = np.array([starts, ends, durations]).T
    blinks = Blinks(blinkarray)
    
    return blinks, signal #return structure containing blink information, and trace that indicates whether a sample was missing or not

def _rename_attribute(obj, old_name, new_name):
    obj.__dict__[new_name] = obj.__dict__.pop(old_name)