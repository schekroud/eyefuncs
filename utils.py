import numpy as np
import scipy as sp
from .epochs import epochedEyes

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

def epochs(data, tmin, tmax, triggers, channels):
    chanlist = channels
    nblocks = data.nblocks
    srate = data.srate
    nchans = len(channels)
    blocks = data.blocks
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
        iepochs = np.zeros(shape = [trigtimes.size, nchans, np.arange(tmin, tmax, 1/srate).size])
        for itrig in range(tmins.size):
            for ichan in range(nchans):
                iepochs[itrig, ichan] = getattr(tmpdata, channels[ichan])[tmins[itrig]:tmaxs[itrig]]
        allepochs.append(iepochs)
        alltrigs.append(epoched_events)
    stacked  = np.vstack(allepochs)
    alltrigs = np.hstack(alltrigs)
    epochtimes = np.arange(tmin, tmax, 1/srate)
    #round this to match the sampling rate
    epochtimes = np.round(epochtimes, 3) #round to the nearest milisecond as we dont record faster than 1khz
    
    #create new object
    epoched = epochedEyes(data = stacked, srate = srate, events = alltrigs, times = epochtimes, channels = chanlist)
    setattr(epoched, 'blocks', blocks) #log the blocks of data that went into this epoched structure
    # setattr(epoched, channels, chanlist)

    return epoched
