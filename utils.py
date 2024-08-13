import numpy as np
import scipy as sp
import pickle
from .classes import Blinks
from .raw import rawEyes
from .epoched import epochedEyes

def _calculate_blink_periods(pupil, srate,  blinkspd, maxvelthresh, maxpupilsize, cleanms):
    signal = pupil.copy()
    vel    = np.diff(pupil) #derivative of pupil diameter
    speed  = np.abs(vel)    #absolute velocity
    smoothv   = smooth(vel, twin = 8, method = 'boxcar') #smooth with a 8ms boxcar to remove tremor in signal
    smoothspd = smooth(speed, twin = 8, method = 'boxcar') #smooth to remove some tremor
    #not sure if it quantitatively changes anything if you use a gaussian instead. the gauss filter makes it smoother though
    
    #pupil size only ever reaches zero if missing data. so we'll log this as missing data anyways
    zerosamples = np.zeros_like(pupil, dtype=bool)
    zerosamples[pupil==0] = True
    
    #create an array logging bad samples in the trace
    badsamples = np.zeros_like(pupil, dtype=bool)
    badsamples[1:] = np.logical_or(speed >= maxvelthresh, pupil[1:] > maxpupilsize)
    
    #a quick way of marking data for removal is to smooth badsamples with a boxcar of the same width as your buffer.
    #it spreads the 1s in badsamples to the buffer period around (each value becomes 1/buffer width)
    #can then just check if badsamples > 0 and it gets all samples in the contaminated window
    badsamples = np.greater(smooth(badsamples.astype(float), twin = int(cleanms), method = 'boxcar'), 0).astype(bool)
    badsamps = (badsamples | zerosamples) #get whether its marked as a bad sample, OR marked as a previously zero sample ('blinks' to be interpolated)
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


def smooth(signal, twin = 50, method = 'boxcar'):
    '''

    function to smooth a signal. defaults to a 50ms boxcar smoothing (so quite small), just smooths out some of the tremor in the trace signals to clean it a bit
    can change the following parameters:

    twin    -- number of samples (if 1KHz sampling rate, then ms) for the window
    method  -- type of smoothing (defaults to a boxcar smoothing) - defaults to a boxcar
    '''
    if method == 'boxcar':
        #set up the boxcar
        filt = sp.signal.windows.boxcar(twin)

    #smooth the signal
    if method == 'boxcar':
        smoothed_signal = np.convolve(filt/filt.sum(), signal, mode = 'same')

    return smoothed_signal

def strip_plr(data, plrtrigger, pre_buffer = 3):
    if type(data) != rawEyes:
        raise Exception(f'type must be rawEyes, not {type(data)}')
        
    for iblock in range(data.nblocks):
        if plrtrigger in data.data[iblock].triggers.event_id:
            tmpdata = data.data[iblock]
            plrtrigs = np.where(tmpdata.triggers.event_id == plrtrigger)[0] #get indices of plr triggers
            ftrig = plrtrigs[-1]+1 #get the next trigger after the last PLR (start of the first trial of task)
            ftrig_time = tmpdata.triggers.timestamp[ftrig]
            ftrigtime_cropped = ftrig_time - (data.srate*pre_buffer)
            
            #find all timepoints that occur before this cropped timepoint
            delinds = np.squeeze(np.where(tmpdata.trackertime < ftrigtime_cropped))
            
            #remove data from relevant signals
            tmpdata.xpos = np.delete(tmpdata.xpos, delinds)
            tmpdata.ypos = np.delete(tmpdata.ypos, delinds)
            tmpdata.pupil = np.delete(tmpdata.pupil, delinds)
            tmpdata.trackertime = np.delete(tmpdata.trackertime, delinds)
            tmpdata.fsamp = tmpdata.trackertime[0] #reset the first sample
            tmpdata.time = np.subtract(tmpdata.trackertime, tmpdata.fsamp) #update the time array
            
            trigs2rem = np.where(tmpdata.triggers.timestamp < ftrigtime_cropped)
            tmpdata.triggers.timestamp = np.delete(tmpdata.triggers.timestamp, trigs2rem)
            tmpdata.triggers.event_id  = np.delete(tmpdata.triggers.event_id, trigs2rem)
            
            #set the data
            data.data[iblock] = tmpdata
    
    #get the first sample again and update if needed
    fsamp = data.fsamp
    newfsamp = data.data[0].trackertime.min()
    if int(fsamp) <= int(newfsamp):
        data.fsamp = newfsamp
    
    return data #return the stripped data object

def interpolate_blinks(data):
    if type(data) != rawEyes:
        raise Exception(f'type must be rawEyes, not {type(data)}')
    
    for iblock in range(data.nblocks):
        tmpdata  = data.data[iblock]
        pupil    = tmpdata.pupil.copy()
        nanpupil = tmpdata.pupil_nan.copy()
        times    = tmpdata.time.copy()
        
        mask = np.zeros_like(times, dtype=bool)
        mask |= np.isnan(nanpupil)
        
        interpolated = np.interp(
            times[mask],
            times[~mask],
            pupil[~mask]
            )
        
        cleanpupil = nanpupil.copy()
        cleanpupil[mask] = interpolated
        data.data[iblock].pupil_clean = cleanpupil
    return data

def epochs(data, tmin, tmax, triggers, channel = 'pupil_clean'):
    nblocks = data.nblocks
    srate = data.srate
    allepochs = []
    alltrigs  = []
    for iblock in range(nblocks):
        tmpdata = data.data[iblock]
        findtrigs = np.isin(tmpdata.triggers.event_id, triggers) #check if triggers are present
        epoched_events = tmpdata.triggers.event_id[findtrigs] #store the triggers that are found, in order
        trigttimes = tmpdata.triggers.timestamp[findtrigs]     #get trackertime for the trigger onset
        trigtimes = np.squeeze(np.where(np.isin(tmpdata.trackertime, trigttimes))) #get indices of the trigger onsets
        tmins = np.add(trigtimes, tmin*srate).astype(int) #enforce integer so you can use it as an index
        tmaxs = np.add(trigtimes, tmax*srate).astype(int) #enforce integer so you can use it as an index
        iepochs = np.zeros(shape = [trigtimes.size, np.arange(tmin, tmax, 1/srate).size])
        for itrig in range(tmins.size):
            iepochs[itrig] = tmpdata.__dict__[channel][tmins[itrig]:tmaxs[itrig]]
        allepochs.append(iepochs)
        alltrigs.append(epoched_events)
    stacked  = np.vstack(allepochs)
    alltrigs = np.hstack(alltrigs)
    epochtimes = np.arange(tmin, tmax, 1/srate)
    #round this to match the sampling rate
    epochtimes = np.round(epochtimes, 3) #round to the nearest milisecond as we dont record faster than 1khz
    
    #create new object
    epoched = epochedEyes(data = stacked, srate = srate, events = alltrigs, times = epochtimes)
    
    return epoched