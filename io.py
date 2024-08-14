import numpy as np
import pickle
from .raw import rawEyes
from .classes import EyeHolder


def parse_eyes(fname, srate = 1000):#, binocular = False):
    # if binocular:
    #     eyedata = _parse_binocular(fname, srate)
    # elif not binocular:
    #     eyedata = _parse_monocular(fname, srate)
    eyedata = _parse_eyes(fname, srate)
    
    return eyedata

def _parse_eyes(fname, srate):
    ncols_search = [6, 9] #if monocular search for 6 columns, if binocular search for 9 columns
    d = open(fname, 'r')
    raw_d = d.readlines()
    d.close()
    starts = [x for x in range(len(raw_d)) if raw_d[x] != '\n' and raw_d[x].split()[0] == 'START']
    fstart = starts[0]
    raw_d = raw_d[fstart:] #remove calibration info at the beginning of the file opening
    
    starts = [x for x in range(len(raw_d)) if raw_d[x] != '\n' and raw_d[x].split()[0] == 'START']
    ends   = [x for x in range(len(raw_d)) if raw_d[x] != '\n' and raw_d[x].split()[0] == 'END']

    # if len(ends) == len(starts)-1:
        # print(f'')
    
    eyedata = rawEyes(nblocks=len(starts), srate = srate)
    eyedata.binocular=True #log that this *is* a binocular recording
    
    # by handling binocular/monocular separately for each block of the data
    # you can handle situations where you change binocular/monocular between task blocks
    
    nsegments = len(starts)
    for iseg in range(nsegments):
        print(f'parsing block {iseg+1}/{nsegments}')
        istart, iend = starts[iseg], ends[iseg]
        rdata = raw_d[istart:iend+1]
        startmsg = rdata[0].split() #parse the start message as this tells you how many eyes are recorded
        if 'LEFT' in startmsg and 'RIGHT' in startmsg:
            binoc = True
            print('-- block recording is binocular')
        else:
            binoc = False
            print('-- block recording is monocular')
        data = rdata[7:] #cut out some of the nonsense before recording starts
        idata = [x for x in data if len(x.split()) == ncols_search[int(binoc)]]
        msgs  = [x for x in data if len(x.split()) != ncols_search[int(binoc)]]
        idata = np.asarray([x.split() for x in idata])[:, :-1] #drop the last column as it is nonsense (it's just '.....')
        if iseg == 0:
            fsamp = int(idata[0][0]) #get starting sample number
            eyedata.fsamp = fsamp
        #missing data is coded as '.' in the asc file - we need to make this usable but easily identifiable
        idata = np.where(idata=='.', 'NaN', idata) #set this missing data to a string nan that numpy can handle easily
        idata = idata.copy().astype(float) #convert into numbers now
        
        segdata             = EyeHolder()
        segdata.binocular   = binoc
        if 'LEFT' in startmsg:
            segdata.eyes_recorded.append('left')
        if 'RIGHT' in startmsg:
            segdata.eyes_recorded.append('right')
        
        segdata.trackertime = idata[:,0]
        segdata.time        = np.subtract(segdata.trackertime, segdata.trackertime[0]) #time relative to the first sample
        
        if segdata.binocular: #if binocular, add both eyes
            colsadd = ['xpos_l', 'ypos_l', 'pupil_l', 'xpos_r', 'ypos_r', 'pupil_r']
            for icol in range(len(colsadd)):
                setattr(segdata, colsadd[icol], idata[:,icol+1])
        if not segdata.binocular : #we'll log which eye was recorded for the block, but going to harmonise this down the line by saving with one name only regardless of eye
            colsadd = ['xpos', 'ypos', 'pupil']
            for icol in range(len(colsadd)):
                setattr(segdata, colsadd[icol], idata[:, icol+1])
        
        #some parsing of triggers etc
        msgs = [x.split() for x in msgs]
        if len([x for x in msgs if len(x)==0])>0: #check if we have any items that are empty and drop if so
            msgs = [x for x in msgs if len(x)>0]
        triggers = np.array([x for x in msgs if x[0] == 'MSG'])
        segdata.triggers.timestamp = triggers[:,1].astype(int)
        segdata.triggers.event_id  = triggers[:,2]
        
        eyedata.data.append(segdata)
    return eyedata

def _parse_monocular(fname, srate):
    #by default it's just going to look for stop/starts in the data file
    d = open(fname, 'r')
    raw_d = d.readlines()
    d.close()
    starts = [x for x in range(len(raw_d)) if raw_d[x] != '\n' and raw_d[x].split()[0] == 'START']
    fstart = starts[0]
    raw_d = raw_d[fstart:] #remove calibration info at the beginning of the file opening
    
    starts = [x for x in range(len(raw_d)) if raw_d[x] != '\n' and raw_d[x].split()[0] == 'START']
    ends   = [x for x in range(len(raw_d)) if raw_d[x] != '\n' and raw_d[x].split()[0] == 'END']
    
    eyedata = rawEyes(nblocks=len(starts), srate = srate)
    eyedata.binocular = False #log that this is *not* a binocular recording
    nsegments = len(starts)
    for iseg in range(nsegments):
        print(f'parsing block {iseg+1}/{nsegments}')
        istart, iend = starts[iseg], ends[iseg]
        data = raw_d[istart:iend+1]
        data = data[7:] #cut out some of the nonsense before recording starts
        idata = [x for x in data if len(x.split()) == 6]
        msgs  = [x for x in data if len(x.split()) != 6]
        
        idata = np.asarray([x.split() for x in idata])[:,:-1] #drop the last column as it is nonsense
        #idata now has: [trackertime, x, y, pupil, something random]
        if iseg == 0:
            fsamp = int(idata[0][0]) #get starting sample number
            eyedata.fsamp = fsamp
        #missing data is coded as '.' in the asc file - we need to make this usable but easily identifiable
        idata = np.where(idata=='.', 'NaN', idata) #set this missing data to a string nan that numpy can handle easily
        idata = idata.copy().astype(float) #convert into numbers now
        
        segdata             = EyeHolder()
        #segdata.fsamp       = fsamp
        segdata.trackertime = idata[:,0]
        segdata.xpos        = idata[:,1]
        segdata.ypos        = idata[:,2]
        segdata.pupil       = idata[:,3]
        segdata.time        = np.subtract(segdata.trackertime, segdata.trackertime[0]) #time relative to the first sample
        
        #some parsing of triggers etc
        msgs = [x.split() for x in msgs]
        triggers = np.array([x for x in msgs if x[0] == 'MSG'])
        segdata.triggers.timestamp = triggers[:,1].astype(int)
        segdata.triggers.event_id  = triggers[:,2]
        
        eyedata.data.append(segdata)
    return eyedata # type: ignore

def save(obj, fname):
        with open(fname, 'wb') as handle:
            pickle.dump(obj, handle)