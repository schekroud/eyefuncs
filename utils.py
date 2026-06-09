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

def _strip_plr(data, blockid, plrtrigger, buffer):
    blockdata = data.data[blockid]
    plrtrigs = np.where(blockdata.triggers.event_id == plrtrigger)[0]
    
    if plrtrigs[0] == 0: #first trigger of the block is the PLR, so the PLR was run before the task
        plrloc = 'start'
        ftrig = plrtrigs[-1]+1 #get the next trigger after the last PLR (start of the first trial of task)
        ftrig_time = blockdata.triggers.timestamp[ftrig]
        ftrigtime_cropped = ftrig_time - (data.srate*buffer) #some buffer to allow measurement of pre-trial pupil size
        #find all timepoints that occur before this cropped timepoint
        delinds = np.squeeze(np.where(blockdata.trackertime < ftrigtime_cropped))
    elif plrtrigs[0] != 0: #if not the first trigger of the block, PLR was run at the end of the task block (i.e. end of the experiment)
        plrloc = 'end'
        ftrig = plrtrigs[0]-1 #get the previous trigger before the PLR started
        ftrig_time = blockdata.triggers.timestamp[ftrig]
        ftrigtime_cropped = ftrig_time + (data.srate*buffer) #some buffer after the last trigger to allow post-trial measurements
        #find all timepoints that occur AFTER this cropped timepoint
        delinds = np.squeeze(np.where(blockdata.trackertime > ftrigtime_cropped))
    
    #remove data from the appropriate channels based on monocular/binocular recordings
    if blockdata.binocular:
        for ieye in blockdata.eyes_recorded:
            for trace in ['xpos', 'ypos', 'pupil']:
                tmp = getattr(blockdata, f'{trace}_{ieye[0]}')
                tmp = np.delete(tmp, delinds)
                setattr(blockdata, f'{trace}_{ieye[0]}', tmp)
    elif not blockdata.binocular:
        for trace in ['xpos', 'ypos', 'pupil']:
            tmp = getattr(blockdata, trace)
            tmp = np.delete(tmp, delinds)
            setattr(blockdata, trace, tmp)
    
    blockdata.trackertime = np.delete(blockdata.trackertime, delinds)
    tmpfsamp = blockdata.trackertime[0] #reset the first sample. if the first sample was unchanged, this just recalculates the same array
    blockdata.time = np.subtract(blockdata.trackertime, tmpfsamp) #update the time array too. if PLR was at the end, this effectively just crops the time array
    
    if plrloc == 'start':
        trigs2rem = np.where(blockdata.triggers.timestamp < ftrigtime_cropped)
    elif plrloc == 'end':
        trigs2rem = np.where(blockdata.triggers.timestamp > ftrigtime_cropped)
    blockdata.triggers.timestamp = np.delete(blockdata.triggers.timestamp, trigs2rem)
    blockdata.triggers.event_id  = np.delete(blockdata.triggers.event_id, trigs2rem)
    
    data.data[blockid] = blockdata
    return data

def strip_plr(data, plrtrigger, buffer = 3):
    '''
    Routine to measure the Pupillary Light Response was performed twice - once before the beginning of the first block, once at the end of the second block
    need to handle these separately as different assumptions.

    NOTE: this only does anything if it finds a block of recording data where a PLR was run, otherwise nothing changes
    '''
    for iblock in range(data.nblocks):
        if plrtrigger in data.data[iblock].triggers.event_id:
            data = _strip_plr(data, iblock, plrtrigger, buffer)    
            
    #get the first sample again and update if needed
    fsamp = data.fsamp
    newfsamp = data.data[0].trackertime.min() #get new first sample time of the data after stripping PLR
    if int(fsamp) <= int(newfsamp): #this is TRUE if the PLR was at the start of the block recording
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
