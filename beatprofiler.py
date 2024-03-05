# Last updated in 02/14/2024 by Youngbin Kim
import sys
import scipy as sp
from scipy import signal
import pandas as pd
import numpy as np
# scikit-video uses deprecated np variables
np.float = np.float64
np.int = np.int_
import glob
import os
import warnings
import bisect
from sklearn.preprocessing import minmax_scale
from sklearn.mixture import GaussianMixture
import yaml
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to prevent GUI-related issues like segmentation faults
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle

class Video():
    '''
    A class used to read videos and extract traces
    
    Attributes
    -----------
    file_path : str
        Path of the file or pycromanager folder
    frame_rate : float
        frame rate of video. optional if acq_mode is pycromanager or filetype is nd2
    max_bpm : int
        max beat rate for the cardiomyocytes. Used in calculate_mask if the method is fundamental. Also used in calculate_reference_frame for BFVideo to calculate minimum peak distance.
    acq_mode : str
        Must specify as 'pycromanager' for Pycromanager acquired tif stacks
        otherwise, it should be None
    name : str
        Name of the video
    mask : np.array
        A boolean mask if you need to process a specifc ROI. Filled with True by default
        shape is W x H
    Methods
    -------
    calculate_mask() : np.array
        2D boolean array of area that contain beating pixels W x H
    '''

    def __init__(self, file_path, frame_rate=None, max_bpm=360, acq_mode = None, low_ram=True, chunk_sec=5, name=None):
        '''
        Read the video and extracts/saves the frame rate

        '''
        self.name = name
        self.max_bpm = max_bpm
        self.trace = None
        self.file_ext = os.path.splitext(file_path)[-1]
        self.frame_rate = frame_rate
        self.acq_mode = acq_mode
        self.file_path = file_path
        self.n_frames = 0
        self.low_ram = low_ram
        self.chunk_size = int(chunk_sec * self.frame_rate) # 5 second chunks for memory efficiency
        
        if not self.file_ext and not self.acq_mode:
            # folder was selected and acquisition type is not pycromanager
            raise Exception("You must select a file rather than a folder if acq_mode is not pycromanager")
        
        if self.acq_mode == 'pycromanager':
            # pycromanager requires file_path to be the parent folder containing 'Full Resolution' folder
            import tifffile

            # if video is more than 4gb, pycromanager automatically splits it into multiple files
            self.img_paths = np.array(glob.glob(os.path.join(self.file_path, "**/*.tif"), recursive=True))
            self.img_paths = np.sort(self.img_paths)
            
            #get n frames
            
            length = 0
            lengths = []
            for file_path in self.img_paths:
                with tifffile.TiffFile(file_path) as tif:
                    cur_length = len(tif.pages)
                lengths.append(cur_length)
                length += cur_length
            self.n_frames = length
            self.first_frame = tifffile.imread(self.img_paths[0], key=0)
            
            # frame by frame reader used for low ram or tissue videos
            def video_reader():
                for file_path in self.img_paths:
                    with tifffile.TiffFile(file_path) as tif:
                        for frame in tif.pages:
                            yield frame.asarray()
            self.video_reader = video_reader
                
            if not self.low_ram and not isinstance(self, TissueVideo):
                self.raw_video = tifffile.imread(self.img_paths[0])
                for i in range(1, len(self.img_paths)):
                    self.raw_video = np.concatenate([self.raw_video, tifffile.imread(self.img_paths[i])])
            
            # extract frame rate if not specified
            if not self.frame_rate:
                from pycromanager import Dataset
                # extract metadata using pycromanager class
                # because raw data handling is super slow we load the whole thing to RAM by calling np.asarray() (good for computational speed)
                # while pycromanager loads into dask format, which loads each frame as needed (good for low RAM)
                dataset = Dataset(self.file_path)
                self.frame_rate = float(dataset.read_metadata(time=0)['Andor sCMOS Camera-FrameRate'])
        
        elif self.file_ext == '.nd2':
            from nd2reader import ND2Reader

            nd2vid = ND2Reader(self.file_path)
            self.n_frames = len(nd2vid)
            self.first_frame = np.array(ND2Reader(self.file_path)[0])
            # extract frame rate if not specified
            if not self.frame_rate:
                self.frame_rate = nd2vid.frame_rate
            # frame by frame reader used for low ram or tissue videos
            def video_reader():
                n = 0
                while n < self.n_frames:
                    yield nd2vid[n]
                    n +=1
            self.video_reader = video_reader
            if not self.low_ram and not isinstance(self, TissueVideo):
                self.raw_video = np.array(nd2vid)

        elif self.file_ext == '.tif':
            import tifffile
            with tifffile.TiffFile(self.file_path) as tif:
                self.n_frames = len(tif.pages)
            self.first_frame = tifffile.imread(self.file_path, key=0)
            self.frame_rate = frame_rate
            
            # frame by frame reader used for low ram or tissue videos
            def video_reader():
                with tifffile.TiffFile(self.file_path) as tif:
                    for frame in tif.pages:
                        yield frame.asarray()
            self.video_reader = video_reader
            if not self.low_ram and not isinstance(self, TissueVideo):
                self.raw_video = tifffile.imread(self.file_path)
                
        else: # mp4, avi, etc. use skvideo to import
            ## need to set FFMPEG path for GUI version
            import skvideo
            if hasattr(sys, '_MEIPASS'):
                # PyInstaller creates a temp folder and stores path in _MEIPASS
                base_path = sys._MEIPASS
                skvideo.setFFmpegPath(os.path.join(base_path, "ffmpeg"))
            import skvideo.io
            ffmpegreader = skvideo.io.FFmpegReader(self.file_path)
            self.n_frames = ffmpegreader.getShape()[0]
            self.first_frame = skvideo.io.vread(self.file_path, num_frames=1).squeeze().mean(axis=-1)
            self.frame_rate = frame_rate
            
            # frame by frame reader used for low ram or tissue videos
            # make video black and white
            def video_reader():
                ffmpegreader = skvideo.io.FFmpegReader(self.file_path)
                generator = ffmpegreader.nextFrame()
                for frame in generator:
                    yield frame.mean(axis=-1)
            self.video_reader = video_reader
            if not self.low_ram and not isinstance(self, TissueVideo):
                self.raw_video = skvideo.io.vread(self.file_path).mean(axis=-1)

        self.mask = np.full(self.first_frame.shape, True)
    
    def __len__(self):
        return self.n_frames

    def calculate_mask(self, method=None, mask_file=None, invert=False, savefig=False, savefig_path=None, yolo_seg_model=None):
        '''
        Calculates and saves self.mask

        Parameters
        ----------
        method : str
            Method used to calculate the mask
            None uses all pixels.
            'mean fft' is recommended for most fluorescent videos.
            'dynamic range' is recommended for most brightfield videos and is best for speed.
            'YOLOv8 segmentation' uses a deep learning model to automatically segment a mask for brightfield videos.
            You may need to empirically find which method works best for your videos."
        mask_file : str or None
            path of the mask file if you want to use an existing mask file
        invert : bool
            mask detection is incorrectly inverted sometimes. to invert the mask, chagne this to True.
        savefig : bool
            boolean indicating whether mask should be exported to a file    
        savefig_path : str or None
            file path of the mask image that will be exported if savefig is True

        Returns
        -------
        self.mask : np.array
        shape is W x H
        '''
        if not method:
            return self.mask

        if mask_file:
            mask_img = plt.imread(mask_file)
            if len(mask_img.shape()) > 2:
                mask_img = mask_img.mean(axis=-1)
            self.mask = mask_img.astype(bool)
            return self.mask

        if method not in ["dynamic range", "max fft", "mean fft", "fundamental", "YOLOv8 segmentation"]:
            raise Exception("Invalid filter method. Must be one of 'dynamic range', 'max fft', 'mean fft', 'fundamental', or 'YOLOv8 segmentation'.")
        
        # helper function used in low ram settings to process videos in chunks
        def chunk_agg(fn, agg_fn, chunk_size):
            i=0
            agg_data = None
            video_reader = self.video_reader()
            if chunk_size > len(self):
                video_chunk = np.zeros((len(self), *self.first_frame.shape))
                for n in range(len(self)):
                    video_chunk[n] = next(video_reader)
                agg_data = fn(video_chunk)

            while i < len(self) and i+chunk_size <= len(self):
                video_chunk = np.zeros((chunk_size, *self.first_frame.shape))
                for n in range(chunk_size):
                    video_chunk[n] = next(video_reader)
                test = fn(video_chunk)
                if agg_data is None:
                    agg_data = fn(video_chunk)
                else:
                    agg_data = agg_fn(agg_data, fn(video_chunk))
                i += chunk_size
            return agg_data

        if method == "YOLOv8 segmentation":
            from ultralytics import YOLO
            mask_model = YOLO(yolo_seg_model)
            # increase contrast on image before predicting to improve results
            p2, p98 = np.percentile(self.first_frame, (2, 98))
            img_clipped = np.clip(self.first_frame, p2, p98)
            img_scaled = minmax_scale(img_clipped, feature_range=(0, 255))
            frame = np.stack((img_scaled,)*3, axis=-1)
            results = mask_model.predict(source=frame)
            mask = cv2.resize(np.sum(results[0].masks.data.cpu().numpy(), axis=0), results[0].orig_shape[::-1])
            self.mask = cv2.threshold(mask, thresh=0.5, maxval=1, type=cv2.THRESH_BINARY)[1]
        
        elif method == "dynamic range":
            # high dynamic range is more likely from active regions
            if self.low_ram:
                min_pix = chunk_agg(lambda x: np.min(x, axis=0), lambda x,y: np.min([x,y], axis=0), self.chunk_size)
                max_pix = chunk_agg(lambda x: np.max(x, axis=0), lambda x,y: np.max([x,y], axis=0), self.chunk_size)
                fullrange = max_pix - min_pix
            else:
                fullrange = np.max(self.raw_video,axis=0) - np.min(self.raw_video,axis=0)

            gmm3 = GaussianMixture(n_components=2).fit(fullrange.reshape(-1,1))
            labels3 = gmm3.predict(fullrange.reshape(-1,1)).reshape(self.first_frame.shape)
            signal_group = np.argmax(gmm3.means_)
            self.mask = labels3 == signal_group

        elif method == "max fft":
            # filter #1 is based on max fft amplitude
            # the distribution is bimodal with the signal being the gaussian with the higher center
            # noise tends to have lower max fft amplitude (but not always, which is why we have the invert option)
            # sometimes this filter doesnt work too well if video is noisy (they are sometimes inverted or there are multiple peaks)
            if self.low_ram:
                sum_fft = chunk_agg(lambda x: np.abs(np.fft.fft(x, axis=0)), lambda x,y: np.sum([x,y],axis=0), self.chunk_size)
                max_fft = np.max(sum_fft[1:,:,:], axis=0)
            else:
                max_fft = np.max(np.abs(np.fft.fft(np.array(self.raw_video), axis=0)[1:,:,:]), axis=0)

            gmm = GaussianMixture(n_components=2).fit(max_fft.reshape(-1,1))
            labels = gmm.predict(max_fft.reshape(-1,1)).reshape(self.first_frame.shape)
            signal_group = np.argmax(gmm.means_)
            self.mask = labels == signal_group

        elif method == "mean fft":
            # filter 1a is based on mean fft amplitude
            if self.low_ram:
                sum_fft = chunk_agg(lambda x: np.abs(np.fft.fft(x, axis=0)), lambda x,y: np.sum([x,y],axis=0), self.chunk_size)
                mean_fft = np.mean(sum_fft, axis=0)
            else:
                mean_fft = np.mean(np.abs(np.fft.fft(np.array(self.raw_video), axis=0)), axis=0)
            gmm1a = GaussianMixture(n_components=2).fit(mean_fft.reshape(-1,1))
            labels1a = gmm1a.predict(mean_fft.reshape(-1,1)).reshape(self.first_frame.shape)
            signal_group = np.argmax(gmm1a.means_)
            self.mask = labels1a == signal_group

        elif method == "fundamental":
            # filter #2 is based on fundamental frequency of each pixel
            # it filters wihtin the physiological range only (less than 360bpm)
            # subtract average value by removing fft[0,:,:]. had to add back 1 because we truncated the array
            if self.low_ram:
                fft = chunk_agg(lambda x: np.abs(np.fft.fft(x, axis=0)), lambda x,y: np.sum([x,y],axis=0), self.chunk_size)
                argmax_fft = np.argmax(sum_fft[1:,:,:], axis=0)+1
            else:
                fft = np.fft.fft(np.array(self.raw_video), axis=0)
                argmax_fft = np.argmax(np.abs(fft[1:,:,:]), axis=0)+1
            # not the best for noisy data
            min_rr_frame = (self.max_bpm / 60) * len(fft) / self.frame_rate # for example, 250bpm is 4.17 Hz which is 240ms between beats. this * fps =  min # frames between beats
            labels2 = (argmax_fft < min_rr_frame) | (argmax_fft > np.max(fft)-min_rr_frame)
            self.mask = labels2
        
        # flip mask if invert
        if invert:
            self.mask = np.logical_not(self.mask)

        # export mask if specified
        if savefig is True:
            self.save_mask(savefig_path)

        self.area = self.mask.sum()
        return self.mask

    def save_mask(self, savefig_path):
        assert savefig_path is not None, "savefig_path cannot be None if savefig is True"
        os.makedirs(savefig_path, exist_ok=True)
        plt.imsave(fname=os.path.join(savefig_path, self.name.replace("\\", "-")+" mask.png"),
            arr=self.mask, 
            cmap="gray")

    def downsample(arr, new_shape):
        # crops a part of image if simple downsampling ratio is off
        d_shape = [arr.shape[1] // new_shape[0], arr.shape[2] // new_shape[1]]
        shape = (arr.shape[0],
                 new_shape[0], d_shape[0],
                 new_shape[1], d_shape[1])
        return arr[:,:d_shape[0]*new_shape[0], :d_shape[1]*new_shape[1]].reshape(shape).mean(4).mean(2)


class BFVideo(Video):
    '''
    Child class of Video dedicated for brightfield videos
    Read comments in Video to learn about inherited attributes and methods
    
    Attributes
    ----------
    reference_frame : np.array
        Brightfield videos require a reference frame to subtract from to look at pixels that are deviating
        frame shape is W x H
    trace : np.array
        trace from video calculated by averaging the

    Methods
    -------
    calculate_trace(str reference) : np.array
        calculates the reference frame by averaging frames of relaxed states
    calculate_trace(str reference) : np.array
        calculates average change in pixel intensity over time
        using a reference frame as defined in initialization and updates self.trace
    
    '''

    def __init__(self, file_path, frame_rate=None, max_bpm=360, acq_mode = None, low_ram=True, chunk_sec=5, name=None):
        super().__init__(file_path=file_path, frame_rate=frame_rate, max_bpm=360, acq_mode=acq_mode, low_ram=low_ram, chunk_sec=chunk_sec, name=name)
        self.reference_frame = None

    def calculate_reference_frame(self, prominence=0.5, rel_height=0.05):
        '''
        Calculates reference frame
        1. analyze video with mean of first chunk as temp reference frame and identify peaks (most contracted state) - contractile peaks are pretty robust between different reference frame choices
        2. Using the average of peaks as reference, find the inverted trace. Then find peaks to identify the relaxed states
        3. select frames around the new peaks of the inverted trace according to the rel_height tolerance, and avg them as reference frame

        Parameters
        ----------
        prominence : float
            Used to find peaks that have prominence greater than this value
        rel_height : float
            threshold to use when selecting frames for reference. Default is 0.05 meaning any points within 5% of the peak prominence is considered to be valleys (relaxed states) and averaged to find the reference frame

        Returns
        -------
        self.reference_frame : np.array
        shape is W x H

        '''

        ### Step 1
        n_frames = min(len(self), self.chunk_size)
        if self.low_ram:
            video_reader = self.video_reader()
            temp_reference_frame = np.zeros_like(self.first_frame, dtype=float)
            for _ in range(n_frames):
                frame = next(video_reader) * self.mask
                temp_reference_frame += frame / n_frames

            video_reader = self.video_reader()
            temp_trace = np.zeros(len(self))
            for i in range(len(self)):
                temp_trace[i] = np.mean(np.abs(next(video_reader)*self.mask - temp_reference_frame))
            temp_trace = minmax_scale(temp_trace)
        else:
            masked_video = self.raw_video * self.mask
            temp_reference_frame = np.mean(masked_video[:n_frames], axis=0)
            temp_trace = minmax_scale(np.mean(np.abs(masked_video - temp_reference_frame), axis=(1,2)))

        min_RR_index = 1 /(self.max_bpm / 60) * self.frame_rate 
        peaks, properties = sp.signal.find_peaks(temp_trace, distance = max(2,min_RR_index), prominence=prominence, width=0, rel_height =1)


        ### Step 2
        
        # if no peaks use mean as reference
        if len(peaks) == 0:
            if self.low_ram:
                video_reader = self.video_reader()
                self.reference_frame = np.zeros_like(self.first_frame, dtype=float)
                for _ in range(len(self)):
                    frame = next(video_reader) * self.mask
                    self.reference_frame += frame / n_frames
                return self.reference_frame
            else:
                self.reference_frame = np.mean(self.raw_video, axis=0) * self.mask
                return self.reference_frame
        
        # otherwise, use the peaks as temp_reference 2
        if self.low_ram:
            video_reader = self.video_reader()
            temp_reference_frame_2 = np.zeros_like(self.first_frame, dtype=float)
            for i in range(len(self)):
                if i in peaks:
                    temp_reference_frame_2 += next(video_reader) * self.mask / len(peaks)
                else:
                    next(video_reader)

            video_reader = self.video_reader()
            temp_trace_2 = np.zeros(len(self))
            for i in range(len(self)):
                temp_trace_2[i] = np.mean(np.abs(next(video_reader)*self.mask - temp_reference_frame_2))
            temp_trace_2 = minmax_scale(temp_trace_2)
        else:
            temp_reference_frame_2 = np.mean(masked_video[peaks],axis=0)
            temp_trace_2 = minmax_scale(np.mean(np.abs(masked_video - temp_reference_frame_2), axis=(1,2)))


        ### Step 3
        valleys, valley_properties = sp.signal.find_peaks(temp_trace_2, distance = max(2,min_RR_index), prominence=min(prominence, 0.3), width=0, rel_height=rel_height)

        reference_indices = np.array([])

        for i in range(len(valleys)):
            left_i, right_i = np.ceil(valley_properties['left_ips'][i]).astype(int), np.floor(valley_properties['right_ips'][i]).astype(int)

            reference_indices = np.append(reference_indices, range(left_i, right_i+1))

        # get rid of negative indices and drop duplicates
        # make into integer type
        reference_indices = np.unique(reference_indices[reference_indices > 0]).astype(int)

        if self.low_ram:
            video_reader = self.video_reader()
            self.reference_frame = np.zeros_like(self.first_frame, dtype=float)
            for i in range(len(self)):
                if i in reference_indices:
                    self.reference_frame += next(video_reader) * self.mask / len(reference_indices)
                else:
                    next(video_reader)
        else:
            self.reference_frame = np.mean(masked_video[reference_indices], axis=0)

        ## edge case for when there are no valleys found: revert to mean calculation
        if np.isnan(self.reference_frame).sum() > 0:
            if self.low_ram:
                video_reader = self.video_reader()
                self.reference_frame = np.zeros_like(self.first_frame, dtype=float)
                for _ in range(len(self)):
                    self.reference_frame = next(video_reader) * self.mask / len(self)
            else:
                self.reference_frame = np.mean(self.raw_video, axis=0)

        # # for debugging
        # print(self.name)
        # plt.plot(temp_trace)
        # plt.scatter(peaks, temp_trace[peaks], label="peaks")
        # plt.scatter(valleys, temp_trace[valleys], label="valleys")
        # plt.scatter(reference_indices, temp_trace[reference_indices], label="reference_frames")
        # plt.legend()
        # plt.show()

        # plt.plot(temp_trace_2)
        # plt.scatter(peaks, temp_trace_2[peaks], label="peaks")
        # plt.scatter(valleys, temp_trace_2[valleys], label="valleys")
        # plt.scatter(reference_indices, temp_trace_2[reference_indices], label="reference_frames")
        # plt.legend()
        # plt.show()

        return self.reference_frame
        

    def calculate_trace(self, reference = "auto", ref_frame_range = [0,20], reference_frame=None, grid=False):
        '''
        Calculates pixel intensity changes using a user defined reference baseline frame

        Parameters
        ----------
        reference : str
            Method for selecting baseline frame.
            'mean' averages all frames in the video and uses that as baseline frame
            'range' averages defined range of frames and uses that as baseline frame. The frame index range is defined in parameter 'frames'
            'auto' automatic calculation of reference frame. See calculate_reference_frame for more description
            'manual' manual input of reference frame
        ref_frame_range : list
            Only required if reference is 'range'. Frame range to calculate the baseline frame. The cardiomyocytes should be relaxed in this range.
            ref_frames should be 2 element integer list
        reference_frame : np.array or None
            only used when reference is 'manual'
        grid : tuple or None
            if you want to analyze each grid independently (e.g. downsampling or binning), you can input the shape of the output array here
            tuple must be 2 element integers smaller than video dimensions

        Return
        ------
        self.trace : np.array
            Calculated cardiac contractility trace based on pixel intensity changes
        '''
        if reference == "auto":
            self.calculate_reference_frame()
        elif reference == "mean":
            if self.low_ram:
                video_reader = self.video_reader()
                self.reference_frame = np.zeros_like(self.first_frame, dtype=float)
                for _ in range(len(self)):
                    self.reference_frame = next(video_reader) * self.mask / len(self)
            else:
                self.reference_frame = np.mean(self.raw_video, axis=0)
        elif reference == "range":
            if self.low_ram:
                video_reader = self.video_reader()
                self.reference_frame = np.zeros_like(self.first_frame, dtype=float)
                for i in range(0,ref_frame_range[1]):
                    if i >= ref_frame_range[0]:
                        self.reference_frame = next(video_reader) * self.mask / (ref_frame_range[1]-ref_frame_range[0])
                    else:
                        next(video_reader)
            else:
                self.reference_frame = np.mean(self.raw_video[ref_frame_range[0]:ref_frame_range[1]], axis=0)
        elif reference == "manual":
            self.reference_frame = reference_frame
        else:
            raise Exception("Reference type must be 'mean' or 'range'")

        fraction_masked = np.sum(self.mask) / self.mask.size
        
        if self.low_ram:
            video_reader = self.video_reader()
            self.trace = np.zeros(len(self))
            for i in range(len(self)):
                self.trace[i] = np.mean(np.abs(next(video_reader) - self.reference_frame) * self.mask) / fraction_masked
        else:
            self.trace = np.mean(np.abs((self.raw_video - self.reference_frame) * self.mask), axis=(1,2)) / fraction_masked
            
#         if grid:
#             self.grid_trace = super().downsample(np.abs(self.raw_video - self.reference_frame), grid)
#             return self.grid_trace

        return self.trace


class FluoVideo(Video):
    '''
    Child class of Video dedicated for fluorescent videos
    Read comments in Video to learn about inherited attributes and methods

    Methods
    -------
    calculate_trace() : np.array
        calculates average pixel intensity over time and updates self.trace
        shape is F
    calculate_mask() : np.array
        calculates average pixel intensity over time and updates self.trace
        shape is W x H
    
    '''
    def __init__(self, file_path, frame_rate=None, max_bpm=360, acq_mode = None, low_ram=True, chunk_sec=5, name=None):
        super().__init__(file_path=file_path, frame_rate=frame_rate, max_bpm=360, acq_mode=acq_mode, low_ram=low_ram, chunk_sec=chunk_sec, name=name)

    def calculate_trace(self, grid=False):
        '''
        Calculates average pixel intensity over time

        Parameters
        ----------
        use_mask : bool
            boolean indicating whether mask/region of interest should be used to calculate the trace
        grid : tuple or None
            if you want to analyze each grid independently (e.g. downsampling or binning), you can input the shape of the output array here
            tuple must be 2 element integers smaller than video dimensions

        Return
        ------
        self.trace : np.array
            Calculated cardiac contractility trace based on pixel intensity changes
        '''
        fraction_masked = np.sum(self.mask) / self.mask.size
        if self.low_ram:
            video_reader = self.video_reader()
            self.trace = np.zeros(len(self))
            for i in range(len(self)):
                self.trace[i] = np.mean(next(video_reader) * self.mask) / fraction_masked
        else:
            self.trace = np.mean(self.raw_video * self.mask, axis=(1,2)) / fraction_masked
            

#         if grid:
#             if use_mask:
#                 # downsample the mask to rescale the trace properly
#                 mask_scaler = super().downsample(np.reshape(self.mask, [1,*self.mask.shape]), grid).reshape(grid)
#                 self.grid_trace = super().downsample(masked, grid) / mask_scaler
#             else:
#                 self.grid_trace = super().downsample(masked, grid)
#             return self.grid_trace           
        return self.trace

class TissueVideo(Video):
    '''
    Child class of Video dedicated for tissue videos to extract force and displacement
    Read comments in Video to learn about inherited attributes and methods
    
    Attributes
    ----------
    reference_frame : np.array
        Brightfield videos require a reference frame to subtract from to look at pixels that are deviating
        frame shape is W x H
    trace : np.array
        trace from video calculated by averaging the

    Methods
    -------
    calculate_trace(str reference) : np.array
        calculates the reference frame by averaging frames of relaxed states
    calculate_trace(str reference) : np.array
        calculates average change in pixel intensity over time
        using a reference frame as defined in initialization and updates self.trace
    
    '''

    def __init__(self, file_path, frame_rate=None, max_bpm=360, acq_mode = None, low_ram=True, name=None):
        # always low_ram for tissue videos because speed is equivalent
        super().__init__(file_path=file_path, frame_rate=frame_rate, max_bpm=360, acq_mode=acq_mode, low_ram=True, name=name)
        self.center = np.zeros((len(self), 2, 2),dtype=np.float32) # opencv requires float32
        self.bbox = None
        self.width_coord = None
        self.width = None
        self.area = None
        self.resting_dist = None

    def autodetect_pillars(self, yolo_model, frame_i=0):
        '''
        Calculates reference frame
        1. analyze video with mean as temp reference frame and identify peaks (most contracted state) - contractile peaks are pretty robust between different reference frame choices
        2. Using the average of peaks as reference, find the inverted trace. Then find peaks to identify the relaxed states
        3. select frames around the new peaks of the inverted trace according to the rel_height tolerance, and avg them as reference frame

        Parameters
        ----------
        prominence : float
            Used to find peaks that have prominence greater than this value
        rel_height : float
            threshold to use when selecting frames for reference. Default is 0.05 meaning any points within 5% of the peak prominence is considered to be valleys (relaxed states) and averaged to find the reference frame

        Returns
        -------
        self.reference_frame : np.array
        shape is W x H

        '''

        # 1. use yolo for first frame to get center and bounding box
        self.bbox = np.zeros((2, 4))

        # increase contrast on image before predicting to improve results
        p2, p98 = np.percentile(self.first_frame, (2, 98))
        img_clipped = np.clip(self.first_frame, p2, p98)
        img_scaled = minmax_scale(img_clipped, feature_range=(0, 255))
        frame = np.stack((img_scaled,)*3, axis=-1)

        results = yolo_model.predict(source=frame)

        # to do if 0 or 1 boxes are detected
        if len(results[0].boxes.xyxy) < 2:
            raise Exception("Less the 2 pillars are detected.")

        # opencv self.bbox format is different from yolo
        # self.bbox is in (x1,y1,w,h) format where x1, y1 is the coord for the TOP LEFT (NOT center) and w,h are width and height
        if results[0].boxes.xywh[0,0] > results[0].boxes.xywh[1,0]:
            self.center[0, 0] = results[0].boxes.xywh[1, :2].cpu()
            self.center[0, 1] = results[0].boxes.xywh[0, :2].cpu()
            self.bbox[0] = results[0].boxes.xyxy[1].cpu()
            self.bbox[1] = results[0].boxes.xyxy[0].cpu()
        else:
            self.center[0, 0] = results[0].boxes.xywh[0, :2].cpu()
            self.center[0, 1] = results[0].boxes.xywh[1, :2].cpu()
            self.bbox[0] = results[0].boxes.xyxy[0].cpu()
            self.bbox[1] = results[0].boxes.xyxy[1].cpu()

        return self.bbox
    
    def autodetect_points2track(self, n=9):
        assert (self.bbox is not None), "self.bbox must be defined"
        assert self.bbox.shape == (2,4), "self.bbox must have a shape of (2,4)"
        
        # 2. find good features in each pillar head to track using ShiTomasi method

        # increase contrast on image before predicting to improve results
        p2, p98 = np.percentile(self.first_frame, (2, 98))
        img_clipped = np.clip(self.first_frame, p2, p98)
        frame_mono = minmax_scale(img_clipped, feature_range=(0, 255)).astype(np.uint8)
        
        # define masks for each pillar head
        mask1 = np.zeros_like(frame_mono)
        x1, x2, y1, y2 = np.array([self.bbox[0,0], self.bbox[0,2], self.bbox[0,1], self.bbox[0,3]]).astype(int)
        mask1[y1:y2,x1:x2] = 255

        mask2 = np.zeros_like(frame_mono)
        x1, x2, y1, y2 = np.array([self.bbox[1,0], self.bbox[1,2], self.bbox[1,1], self.bbox[1,3]]).astype(int)
        mask2[y1:y2,x1:x2] = 255


        # find points to track within mask (pillar head bounding box)
        self.points1 = cv2.goodFeaturesToTrack(frame_mono, maxCorners=n, qualityLevel=0.1, minDistance=10, blockSize=5, mask=mask1).reshape(-1,2)
        self.points2 = cv2.goodFeaturesToTrack(frame_mono, maxCorners=n, qualityLevel=0.1, minDistance=10, blockSize=5, mask=mask2).reshape(-1,2)
        # find offset for each point 
        self.offset1 = self.points1 - self.center[0, 0]
        self.offset2 = self.points2 - self.center[0, 1]
        
        return self.points1, self.points2
    
    def grid_points2track(self, n=9):
        assert (self.bbox is not None), "self.bbox must be defined"
        assert self.bbox.shape == (2,4), "self.bbox must have a shape of (2,4)"
        
        sqrt_n = int(np.sqrt(n))
        
        spacing1 = ((self.bbox[0,2:] - self.bbox[0,:2])/(sqrt_n+1)).astype(int)
        spacing2 = ((self.bbox[1,2:] - self.bbox[1,:2])/(sqrt_n+1)).astype(int)

        # Create an array of x and y coordinates for a nxn grid 
        x = np.linspace(1, sqrt_n, sqrt_n)
        y = np.linspace(1, sqrt_n, sqrt_n)
        X, Y = np.meshgrid(x, y)
        self.points1 = np.vstack([X.ravel(), Y.ravel()]).T * spacing1 + self.bbox[0,:2]
        self.points2 = np.vstack([X.ravel(), Y.ravel()]).T * spacing2 + self.bbox[1,:2]
        # find offset for each point 
        self.offset1 = self.points1 - self.center[0, 0]
        self.offset2 = self.points2 - self.center[0, 1]
        return self.points1, self.points2

    def calculate_trace(self, bounding_box, anchor_method="auto", n=25, window_size=30, savevid=False, savevid_path=None):
        '''
        Calculates pixel intensity changes using a user defined reference baseline frame

        Parameters
        ----------
        method : str
            Method for tracking movement of defined area or points
            'optical flow' : fast and accurate method for tracking points using Lucas Kanade sparse optical flow method in opencv
            'corr tracker' : slower but accurate method for tracking bounding box using dlib's correlation tracker (variant of MOSSE)
        bounding_box : str or np.array
            Method for selecting anchor points to track.
            'pt_file_path' uses YOLOv8 model to detect anchor bounding box
            'csv_file_path'
            np.array with shape 2x4 uses pixel coordinates to define bounding boxes. Format is (x1,y1,x2,y2) for each box
        anchor_method : str
            Method for selecting anchor points to track.
            'auto' uses ShiTomasi method in opencv to identify n best points to track for each bounding box
            'grid' calculates coordinates for n points in a grid fashion (sqrt_n x sqrt_n) around the center of the bounding box. if n is not a perfect square, it rounds down to the nearest one 
            None is used when method doesnt need anchor points to track (dlib correlation tracker method)
        n: int
            number of points for each anchor to use to track
      

        Return
        ------
        self.trace : np.array
            Calculated cardiac contractility trace based on pixel intensity changes
        '''
        # define bounding box to track distance between
        if type(bounding_box) == np.array:
            assert(bounding_box.shape == (2,4)), "Shape of bounding box array must be (2,4)."
            self.bbox = bounding_box
        elif bounding_box[-3:] == ".pt":
            from ultralytics import YOLO
            model = YOLO(bounding_box)
            self.autodetect_pillars(model, frame_i=0)
        elif bounding_box[-4:] == ".csv":
            self.bbox = pd.read_csv(bounding_box, index_col=0).loc[self.name].to_numpy().reshape(2,4)
        else:
            raise Exception("bounding_box must be a YOLOv8 .pt model, a bounding box .csv file, or (2,4) numpy array.")
            
        if anchor_method == "auto":
            self.autodetect_points2track(n=n)
        elif anchor_method == "grid":
            self.grid_points2track(n=n)
        else:
            raise Exception("Invalid 'anchor_method'. Must be either 'auto' or 'grid'.")

        # create points array to track these points shape (t: number of frames, n:number of points for each pillar x 2, x/y:2)
        points = np.zeros((len(self), (len(self.points1)+len(self.points2)), 2), dtype=np.float32)
        points[0] = np.concatenate((self.points1, self.points2))

        # increase contrast on image before predicting to improve results
        p2, p98 = np.percentile(self.first_frame, (2, 98))
        img_clipped = np.clip(self.first_frame, p2, p98)
        img_scaled = minmax_scale(img_clipped, feature_range=(0, 255)).astype(np.uint8)
        
        
        video_reader = self.video_reader()
        next(video_reader)
        old_frame = np.stack((img_scaled,)*3, axis=-1)

        # Loop through frames in stack
        for i in range(1, len(self)):
            # Get current frame
            unprocessed = next(video_reader)
            p2, p98 = np.percentile(unprocessed, (2, 98))
            img_clipped = np.clip(unprocessed, p2, p98)
            img_scaled = minmax_scale(img_clipped, feature_range=(0, 255)).astype(np.uint8)
            new_frame = np.stack((img_scaled,)*3, axis=-1)

            points[i],status,err = cv2.calcOpticalFlowPyrLK(old_frame, 
                                                            new_frame, 
                                                            prevPts=points[0].reshape(1,-1,2), 
                                                            nextPts=None, 
                                                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                                 100, 0.001),
                                                            winSize=(window_size,window_size),
                                                            maxLevel=4,
                                                           )  

        displacement1 = points[:,:len(self.points1), :] - points[0,:len(self.points1), :]
        displacement2 = points[:,len(self.points1):, :] - points[0,len(self.points1):, :]
        
        # Steps to remove noisy points (poorly tracked points)
        # 1. minmax_scale each point and calculate the std. 
        # 2. remove points that are outliers of the std distribution

        norm_disp1 = [minmax_scale(np.linalg.norm(displacement1[:,pt_i],axis=1)) for pt_i in range(displacement1.shape[1])]
        norm_disp2 = [minmax_scale(np.linalg.norm(displacement2[:,pt_i],axis=1)) for pt_i in range(displacement2.shape[1])]
        
        std1 = np.std(norm_disp1, axis=1)
        std2 = np.std(norm_disp2, axis=1)
        
        def find_outliers_iqr(data):
            """
            Identifies outliers in a dataset using the IQR method.
            Returns a list of indices corresponding to the outlier values.
            """
            quartiles = np.percentile(data, [25, 75])
            iqr = quartiles[1] - quartiles[0]
            lower_bound = quartiles[0] - 1.5 * iqr
            upper_bound = quartiles[1] + 1.5 * iqr
            outlier_indices = []
            for i, x in enumerate(data):
                if x < lower_bound or x > upper_bound:
                    outlier_indices.append(i)
            return outlier_indices
        
        to_drop1 = find_outliers_iqr(std1)
        to_drop2 = find_outliers_iqr(std2)
        

        displacement1 = np.delete(displacement1, to_drop1, axis=1)
        displacement2 = np.delete(displacement2, to_drop2, axis=1)
        
        #shape (#frames,#points,2)
        #self.tracked_points = points
        self.tracked_points = np.delete(points, np.append(to_drop1, np.array(to_drop2)+len(self.points1)).astype(int), axis=1)
            
        self.points1 = np.delete(self.points1, to_drop1, axis=0)
        self.points2 = np.delete(self.points2, to_drop2, axis=0)

        self.center[:, 0] = np.mean(displacement1 + self.center[0, 0], axis=1)
        self.center[:, 1] = np.mean(displacement2 + self.center[0, 1], axis=1)
        self.trace = np.linalg.norm(self.center[:,0] - self.center[:,1], axis=1)

        self.resting_dist = self.trace.max()

        if savevid:
            self.save_labeled_video(savevid_path)
            
        return self.trace
    


    def calculate_width(self, mask=None, anchor_centers=None, savefig=False, savefig_path=None):
        # 1. find midpoint and the slope of the line connecting the two pillar heads
        # 2. use the negative reciprocal and the midpoint to find the equation for the perpendicular line intersecting the midpoint
        # 3. find the min and max points and increment by 1 for the axis with larger increments
        # 4. fill the values in for those points in a dataframe
        # 5. find the longest consecutive 1 in the frame, which we assume to be the tissue length
        if not mask:
            assert not self.mask.all(), "Mask is not defined."
            mask = self.mask
        if not anchor_centers:
            anchor_centers = self.center[0]
            # make sure that self.center[0] is initialized
            assert not (anchor_centers == 0).all(), "Center points at t=0 are not initialized."
        
        [x1, y1], [x2, y2] = anchor_centers
        # find the slope of the perpendicular line
        perp_slope = -(x2-x1)/(y2-y1)
        # find the midpoint
        midpoint_x, midpoint_y = (x1+x2)/2, (y1+y2)/2
        # create the perpendicular line between (x1, y1) and (x2, y2)
        line = lambda y: midpoint_x + 1/perp_slope*(y-midpoint_y)

        # find the edge points
        min_y = 0
        min_x = line(min_y)
        # if min_x is out of bounds, min_x must be 0
        if min_x < 0:
            min_x = 0
            min_y = midpoint_y + perp_slope*(min_x-midpoint_x)

        max_y = mask.shape[0]
        max_x = line(max_y)
        # if max_x is out of bounds, max_x must be the x maximum
        if max_x > mask.shape[1]:
            max_x = 0
            max_y = midpoint_y + perp_slope*(max_x-midpoint_x)

        line_pixel = pd.DataFrame()
        # if perp_slope has magnitude>1 we can increment y by 1, but if its' <1, we should increment by m
        if abs(perp_slope) > 1:
            line_pixel['Y'] = np.arange(min_y, max_y, 1)
        else:
            line_pixel['Y'] = np.arange(min_y, max_y, abs(m))
        line_pixel['X'] = line_pixel['Y'].apply(line)

        # check if each X, Y coordinate is background or tissue
        line_pixel["tissue"] = line_pixel.apply(lambda row: mask[int(row['Y']), int(row['X'])] == 1,axis=1)

        def find_longest_consecutive_tissue(df):
            # returns np.array of shape 2x4 ((x1,y1), (x2,y2)) containing the coordinates of points marking tissue width
            df = line_pixel.copy()
            # Resetting the index to create a column
            df.reset_index(inplace=True)
            # check if next one is different from current
            dfBool = df['tissue'] != df['tissue'].shift()
            # cumsum only changes when the tissue type is different from next
            dfCumsum = dfBool.cumsum()
            # grouping by cumsum allows you to separate the line_pixel by tissue or background
            groups = df.groupby(dfCumsum)
            # for g in groups: print(g)
            # groupcounts gets the min and max index of each contiguous segment and its type
            groupCounts = groups.agg({'index':['count', 'min', 'max'], 'tissue':'first'})
            groupCounts.columns = groupCounts.columns.droplevel()
            #print('\n', groupCounts, '\n')
            # we don't care about background. just tissues
            groupCounts = groupCounts[groupCounts["first"]]
            maxCount = groupCounts[groupCounts['count'] == groupCounts['count'].max()]
            #print(maxCount, '\n')
            if maxCount.empty:
                return [None,None,None,None]
            else:
                maxCount = maxCount.iloc[0]
            return np.array([df.loc[int(maxCount["min"]), ["X", "Y"]], df.loc[int(maxCount["max"]), ["X", "Y"]]], dtype=int).flatten()
        width_x1, width_y1, width_x2, width_y2 = find_longest_consecutive_tissue(line_pixel)
        self.width_coord = np.array([[width_x1, width_y1], [width_x2, width_y2]])
        self.width = np.linalg.norm(self.width_coord[0] - self.width_coord[1])
        
        # export if specified
        if savefig is True:
            self.save_labeled_tissue(savefig_path)
        
        return self.width
    
    def save_labeled_tissue(self, savefig_path):
        assert savefig_path is not None, "savefig_path cannot be None if savefig is True"
        plt.imshow(self.first_frame, cmap="gray")
        plt.imshow(self.mask, alpha=0.5, cmap="copper")
        if self.width_coord is not None:
            plt.plot(self.width_coord[:,0], self.width_coord[:,1], color="cyan", label="tissue width")
        
        if self.bbox is not None:
            #add rectangle
            plt.gca().add_patch(Rectangle(self.bbox[0, :2],width=(self.bbox[0, 2]-self.bbox[0, 0]),height=(self.bbox[0, 3]-self.bbox[0, 1]),
                                edgecolor='#BB5566',
                                facecolor='none',
                                lw=1))

            plt.gca().add_patch(Rectangle(self.bbox[1, :2],width=(self.bbox[1, 2]-self.bbox[1, 0]),height=(self.bbox[1, 3]-self.bbox[1, 1]),
                                edgecolor='#004488',
                                facecolor='none',
                                lw=1))
        #add scatter
        plt.scatter(x=self.points1[:,0], y=self.points1[:,1], color="#BB5566", label="tracked points 1")
        plt.scatter(x=self.points2[:,0], y=self.points2[:,1], color="#004488", label="tracked points 2")
        plt.legend()
        
        os.makedirs(savefig_path, exist_ok=True)
        plt.savefig(fname=os.path.join(savefig_path, self.name.replace("\\", "-")+" points.png"))
        plt.clf()
        
    def save_labeled_video(self, savevid_path):
        assert savevid_path is not None, "savevid_path cannot be None if savefig is True"

        os.makedirs(savevid_path, exist_ok=True)
        # print(os.path.join(savevid_path, self.name.replace("\\", "-")+" tracked_points.mp4"))
        # handle ffmpeg path issues
        import skvideo
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
            skvideo.setFFmpegPath(os.path.join(base_path, "ffmpeg"))

        import skvideo.io
        result = skvideo.io.FFmpegWriter(os.path.join(savevid_path, self.name.replace("\\", "-")+" tracked_points.mp4"), 
                                        inputdict={'-r': str(self.frame_rate),}, outputdict={'-r': str(self.frame_rate),})
        
        video_reader = self.video_reader()
        for frame_i in range(len(self)):
            frame = next(video_reader)
            scaled_frame = minmax_scale(frame, feature_range=(0,255)).astype(np.uint8)
            scaled_frame = np.stack((scaled_frame,)*3,axis=-1)
            for point in self.tracked_points[frame_i]:
                cv2.circle(scaled_frame, tuple(point.astype(int)), 5, (255,0,0),-1)
            for center in self.center[frame_i]:
                cv2.circle(scaled_frame, tuple(center.astype(int)), 10, (0,255,0),-1)
            result.writeFrame(scaled_frame)
        result.close()
            
class Trace:
    '''
    A class used to take a raw trace to filter and analyze it
    
    Attributes
    -----------
    raw_data : np.array
        1D array containing input trace data of shape F
    frame_rate : float
        sampling rate of the trace in # of samples per second
    max_bpm : float
        sampling rate of the trace in # of samples per second
    name : str
        name of the trace
    peaks : np.array
        1D array containing the indices of peaks
    valleys : np.array
        1D array containing the indices of valleys used for photobleach correction
    baseline: np.array
        1D array of len(raw_data) containing the baseline photobleach decay
    df_f0 : np.array
        1D array of len(raw_data) containing the normalized photobleach corrected trace calculated by this equation: (F-F0)/F0 where F is raw_data, and F0 is baseline
    normalized : np.array
        df_f0 rescaled such that the minima is 0 and maxima is 1
    drift_corrected : bool
        a boolean variable indicating whether calculate drift has been run
    feature_summary : pd.DataFrame
        summary of various parameters extracted from the peaks
    peak_summary : pd.DataFrame
        summary of various parameters at the peak level
    beat_segments : pd.DataFrame
        segmented trace so that each peak is a row
    
    Methods
    -------
    calculate_drift() : np.array
        photobleach correction where decay is calculated. returns 1D array of baseline
    analyze_peaks() : None
        analyzes peaks to calculate summary metrics, df_f0, etc.
    '''
    def __init__(self, data, frame_rate, max_bpm=360, name=None, skip_first_n_frames=0):
        self.raw_data = data
        self.frame_rate = frame_rate
        self.max_bpm = max_bpm
        self.name = name
        self.skip_first_n_frames = skip_first_n_frames
        self.peaks = None
        self.valleys = None
        self.contraction_speeds_index = None
        self.relaxation_speeds_index = None
        self.baseline = np.zeros(len(self.raw_data))
        self.df_f0 = None
        self.normalized = None
        self.velocity = None
        self.amplitude_unit = None # "dF/F0" if calculate_drift has been run
        self.feature_summary = pd.Series(index=['num beats', 'beat frequency [bpm]', 'RMSSD [s]',
                                                'mean RR interval [s]', 'mean amplitude [dF/F0]', 'mean raw amplitude [a.u.]', 'mean auc [dF/F0*s]',
                                                'mean tau [s]', 'mean fwhm [s]', 'mean contract50 [s]', 'mean relax50 [s]', 'mean fw90m [s]', 'mean contract90 [s]','mean relax90 [s]',
                                                'mean contraction speed [dF/F0/s]', 'mean relaxation speed [dF/F0/s]', 'mean contraction speed [a.u.]', 'mean relaxation speed [a.u.]', 
                                                'SDRR [s]', 'std amplitude [dF/F0]', 'std raw amplitude [a.u.]', 'std auc [dF/F0*s]',
                                                'std tau [s]', 'std fwhm [s]', 'std contract50 [s]', 'std relax50 [s]', 'std fw90m [s]', 'std contract90 [s]', 'std relax90 [s]',
                                                'std contraction speed [dF/F0/s]', 'std relaxation speed [dF/F0/s]', 'std contraction speed [a.u.]', 'std relaxation speed [a.u.]', 
                                                ],
                                         name=self.name,
                                         dtype=float)
        self.peak_summary = pd.DataFrame(columns=['beat index', 'tau [s]', 'fwhm [s]', 'contract50 [s]', 'relax50 [s]', 'fw90m [s]', 'contract90 [s]',
                                                  'relax90 [s]', 'amplitude [dF/F0]', 'raw amplitude [a.u.]', 'auc [dF/F0*s]',
                                                  'contraction speed [dF/F0/s]', 'relaxation speed [dF/F0/s]', 'contraction speed [a.u.]', 'relaxation speed [a.u.]'])
        self.beat_segments = pd.DataFrame()

    def calculate_drift(self, method=None, min_prominence=0.2):
        '''
        Saves self.valleys (points used to fit baseline), self.baseline, self.normalized (minmaxscale: min=0, max=1), and self.df_f0\

        Parameters
        ----------
        method : str or None
            method to use for baseline calculation.
            None: no drift correction
            'power':  power function from paper referenced below
            'exp': exponential decay function
            'linear': linear interpolation (line of best fit)
            'interpolation': aggressive interpolation (i.e. connect the dot method)
        min_prominence : float
            between 0 and 1. minimum prominence for valley finding algorithm

        Returns
        -------
        self.baseline : np.array
        calculated baseline drift of signal
        '''
        if not method:
            return self.baseline

        ### Step 1: find the valley points to fit our baseline

        # find minimum amount of time between beats given the bpm. this * fps =  min # frames between beats
        min_RR_index = 1 /(self.max_bpm / 60) * self.frame_rate 

        # find valleys to use for our baseline fit
        prominence = min_prominence * np.ptp(self.raw_data)
        self.valleys, _ = sp.signal.find_peaks(-self.raw_data, distance = max(2,min_RR_index), prominence=prominence, width=0, rel_height =1)
        # if less than 3 points, can't fit an equation with 3 unknowns
        # so we need to include more points near the baseline
        # this includes every point that's not within 95% of the peaks 
        if len(self.valleys) < 4:
            peaks, properties = sp.signal.find_peaks(self.raw_data, distance = max(2,min_RR_index), prominence=prominence, width=0, rel_height =0.95)
            valleys = np.full(len(self.raw_data), True)
            for i in range(len(peaks)):
                valleys[int(properties['left_ips'][i]+1):int(properties['right_ips'][i])] = False # remove xrange within 95% height of peak
            self.valleys = np.where(valleys)[0]
        

        ### Step 2: use the valley points to fit the baseline using method specified

        if method == "power":
            # this function from this paper fits super well
            # Interpretation of Fluorescence Decays using a Power-like Model
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1303114/
            def baseline_func(t, tau, q, cons):
                return (2-q)/tau*(1-(1-q)*t/tau)**(1/(1-q))+cons
            try:
                baseline_param, _ = sp.optimize.curve_fit(baseline_func, self.valleys, self.raw_data[self.valleys], p0=[-100,-.5,.5],maxfev=100000)
                tau, q, cons = baseline_param
                self.baseline = baseline_func(np.arange(len(self.raw_data)), tau, q, cons)
            
            #if curve doesnt fit, revert to linear fit
            except RuntimeError:
                raise Warning("Baseline power fit failed. Trying linear fit.")
                method = 'linear'
                
        elif method == "exp":
            def exp_func(t, A, tau, beta, cons):
                return A*np.exp(-(t/tau)**(1-beta))+cons
            # initialization constants to help converge
            cons0 = self.raw_data[-1] # last point's y value
            A0 = self.raw_data[0] - cons0 # 1st point's y value - last point's y value
            tau0 = 3500 # empirical 
            beta0 = 0.07 # empirical 
            try:
                baseline_param, _ = sp.optimize.curve_fit(exp_func, self.valleys, self.raw_data[self.valleys], p0=[A0, tau0, beta0, cons0],maxfev=10000000)
                A, tau, beta, cons = baseline_param
                self.baseline = exp_func(np.arange(len(self.raw_data)), A, tau, beta, cons)
            
            #if curve doesnt fit, revert to linear fit
            except RuntimeError:
                raise Warning("Baseline exp fit failed. Trying linear fit.")
                method = "linear"
                
        elif method == 'linear':
            slope, intercept = np.polyfit(self.valleys, self.raw_data[self.valleys], 1)
            self.baseline = slope * np.arange(len(self.raw_data)) + intercept

        elif method =="interpolation":
            self.baseline = np.interp(np.arange(len(self.raw_data)), self.valleys, self.raw_data[self.valleys])
            
            # apply "linear" (i.e. line of best fit) for points outside of valleys
            slope1, intercept1 = np.polyfit(self.valleys[:2], self.raw_data[self.valleys[:2]], 1)
            self.baseline[:self.valleys[0]] = slope1 * np.arange(len(self.raw_data[:self.valleys[0]])) + intercept1
            slope2, intercept2 = np.polyfit(self.valleys[-2:], self.raw_data[self.valleys[-2:]], 1)
            self.baseline[self.valleys[-1]:] = slope2 * (np.arange(self.valleys[-1], len(self.raw_data))) + intercept2

        # Invalid method
        else:
            raise Error("Invalid method. Method must be one of the following: 'power', 'exp', 'linear', or 'interpolation'.")

        # often there are baseline correction artifacts at the 0th sample so replace it with the next one
        self.baseline[0] = self.baseline[1]

        self.normalized = minmax_scale(self.raw_data - self.baseline)
        self.df_f0 = (self.raw_data-self.baseline)/self.baseline
        self.amplitude_unit = "df/F0"
        return self.baseline

    def analyze_peaks(self, min_prominence=0.25, savefig=False, savefig_path=None):
        '''
        Calculates summary and peak_summary. Calculates peaks, df_f0, and normalized along the way.
        Saves plots for the traces if specified.

        Parameters
        ----------
        min_prominence : float
            between 0 and 1. minimum prominence for peak finding algorithm
        savefig : boolean
            if True, saves plot of raw trace along with peaks, valleys, and baseline
            also saves a plot of df_f0 if drift_corrected
        savefig_path : str or None
            path of where the plot is to be saved if savefig

        Returns
        -------
        None
        '''
        # If you want to run analyze_peaks without baseline correction, it calculates self.normalized
        if not self.amplitude_unit:
            self.normalized = minmax_scale(self.raw_data)
            self.df_f0 = self.raw_data
            min_RR_index = 1 /(self.max_bpm / 60) * self.frame_rate 
            self.valleys, _ = sp.signal.find_peaks(-self.normalized, distance = max(2,min_RR_index), prominence=min_prominence, width=0, rel_height =1)

        # Calculate velocity
        self.velocity = np.gradient(self.df_f0) * self.frame_rate

        ######## Step 1: Find peaks

        # find minimum amount of time between beats given the bpm. this * fps =  min # frames between beats
        min_RR_index = 1 /(self.max_bpm / 60) * self.frame_rate 

        # skip first n samples as specified unless the trace is shorter
        if len(self.raw_data) < self.skip_first_n_frames:
            self.skip_first_n_frames = len(self.raw_data) - 1
        self.peaks, _ = sp.signal.find_peaks(self.normalized[self.skip_first_n_frames:], distance = max(2,min_RR_index), prominence=min_prominence, width=0, rel_height = 0.9)
        self.peaks += self.skip_first_n_frames

        # only keep beats that are full cycles unless peaks and valleys are empty.
        # i.e. they are between two valleys

        if (len(self.peaks) != 0) and (len(self.valleys) != 0):
            self.peaks = self.peaks[(self.peaks > self.valleys.min()) & (self.peaks < self.valleys.max())]

        # quit if peaks and valleys are empty
        if (len(self.peaks) == 0) or (len(self.valleys) == 0):
            self.contraction_speeds_index = np.empty(0, dtype=int)
            self.relaxation_speeds_index = np.empty(0, dtype=int)
            if savefig is True:
                self.save_trace(savefig_path)
            return None

        # segment beats
        segments = []
        segment_indices = []
        # select the closest valley point before and after peak and use those indices to segment
        for p in self.peaks:
            bisect_i = bisect.bisect(self.valleys, p)
            start_i = self.valleys[bisect_i-1]
            end_i = self.valleys[bisect_i]
            segment = self.df_f0[start_i:end_i]
            segments.append(segment)
            segment_indices.append([self.name, start_i, end_i])
        self.beat_segments = pd.DataFrame(segments).reset_index(drop=True)
        self.beat_segments = pd.DataFrame(segment_indices, columns=["index", "start", "end"]).join(self.beat_segments)
        self.beat_segments.index.name = "beat"
        self.beat_segments = self.beat_segments.reset_index().set_index(["index","beat", "start", "end"])

        # drop segments with multiple beats
        def multibeat_finder(segments):
            min_RR_index = 1 /(self.max_bpm / 60) * self.frame_rate 
            to_drop = []
            for i in segments.index:
                peaks, _ = sp.signal.find_peaks(minmax_scale(segments.loc[i].iloc[0:]), 
                                      distance = max(2,min_RR_index), prominence=0.25, width=0, rel_height=1)
                if len(peaks) != 1:
                    to_drop.append(i)
            return to_drop
        to_drop = multibeat_finder(self.beat_segments)
        self.beat_segments = self.beat_segments.drop(to_drop)

        ######## Step 2: Save peak summary
        self.peak_summary['beat index'] = self.peaks
        # calculate contraction and relaxation velocity
        width50 = sp.signal.peak_widths(self.normalized, self.peak_summary['beat index'], rel_height=0.5)
        self.peak_summary['fwhm [s]'] = width50[0] / self.frame_rate # full width half max is default. Contains 3 elements: FWHM, left WHM, right WHM
        self.peak_summary['contract50 [s]'] = (self.peak_summary['beat index'] - width50[2]) / self.frame_rate
        self.peak_summary['relax50 [s]'] = (width50[3] - self.peak_summary['beat index']) / self.frame_rate
        
        width90 = sp.signal.peak_widths(self.normalized, self.peak_summary['beat index'], rel_height=0.9)
        self.peak_summary['fw90m [s]']= width90[0] / self.frame_rate # full/left/right width at 90% of peak. Contains 3 elements: FW90M, left W90M, right W90M
        self.peak_summary['contract90 [s]'] = (self.peak_summary['beat index'] - width90[2]) / self.frame_rate
        self.peak_summary['relax90 [s]'] = (width90[3] - self.peak_summary['beat index']) / self.frame_rate

        # calculate tau
        decay_frame = sp.signal.peak_widths(self.normalized, self.peak_summary['beat index'], rel_height=1-np.exp(-1))[3]
        self.peak_summary['tau [s]'] =  (decay_frame - self.peak_summary['beat index']) / self.frame_rate

        # Calculate only if baseline corrected
        prominences = sp.signal.peak_prominences(self.normalized, self.peaks)[0]
        
        # calculate max and min velocity for each beat segment (corresponds to contraction and relaxation speed)
        contraction_speeds = np.zeros(len(self.peaks))
        relaxation_speeds = np.zeros(len(self.peaks))
        self.contraction_speeds_index = np.zeros(len(self.peaks), dtype=int)
        self.relaxation_speeds_index = np.zeros(len(self.peaks), dtype=int)

        for i in range(len(self.beat_segments)):
            velocity_segment = self.velocity[int(self.beat_segments.iloc[i].name[2]): int(self.beat_segments.iloc[i].name[3])]
            contraction_speeds[i] = velocity_segment.max()
            self.contraction_speeds_index[i] = int(self.beat_segments.iloc[i].name[2])+velocity_segment.argmax()
            relaxation_speeds[i] = -velocity_segment.min()
            self.relaxation_speeds_index[i] = int(self.beat_segments.iloc[i].name[2])+velocity_segment.argmin()



        if self.amplitude_unit:
            self.peak_summary['contraction speed [dF/F0/s]'] = contraction_speeds
            self.peak_summary['relaxation speed [dF/F0/s]'] = relaxation_speeds  
            # calculate amplitude. Need to rescale since prominences were calculated with self.normalized
            self.peak_summary['amplitude [dF/F0]'] = prominences * np.ptp(self.df_f0)
            self.peak_summary['raw amplitude [a.u.]'] = prominences * np.ptp(self.raw_data - self.baseline)
            # calculate area under curve in units AU * sec
            auc = []
            for x0,x1 in np.array([width90[2],width90[3]], dtype=int).T:
                auc.append(np.trapz(self.df_f0[x0:x1], dx=1/self.frame_rate))
            self.peak_summary['auc [dF/F0*s]'] = auc

        else:
            self.peak_summary['contraction speed [a.u.]'] = contraction_speeds
            self.peak_summary['relaxation speed [a.u.]'] = relaxation_speeds
            #calculate raw amplitude differently if not drift correctly since there is no self.baseline
            self.peak_summary['raw amplitude [a.u.]'] = prominences * np.ptp(self.raw_data)
        
        # reindex peak summary for organization
        self.peak_summary.index = [self.name] * len(self.peak_summary)


        ### Step 3: Save full trace summary

        self.feature_summary['num beats'] = len(self.peak_summary)

        # calculate bpm 
        if len(self.peak_summary["beat index"]) > 1: # given there are 2 or more peaks
            self.feature_summary['beat frequency [bpm]'] = (len(self.peak_summary["beat index"]) - 1) / (self.peak_summary["beat index"][-1] - self.peak_summary["beat index"][0]) \
                * self.frame_rate * 60 # convert beats per frame to beats per minute
        
        # calculate mean RR interval, RMSSD and pRR50
        rr_interval = np.diff(self.peak_summary["beat index"]) / self.frame_rate
        self.feature_summary['mean RR interval [s]'] = rr_interval.mean()
        self.feature_summary['SDRR [s]'] = rr_interval.std()
        self.feature_summary['RMSSD [s]'] = np.sqrt(np.mean(rr_interval ** 2))
        
        # calculate mean and std summary
        for col in self.peak_summary.columns.drop("beat index"):
            self.feature_summary['mean ' + col] = self.peak_summary[col].mean()
            self.feature_summary['std ' + col] = self.peak_summary[col].std()
            self.feature_summary['min ' + col] = self.peak_summary[col].min()
            self.feature_summary['max ' + col] = self.peak_summary[col].max()
        
        # if units are not dF/F0 need to handle name changes
        if self.amplitude_unit:
            self.feature_summary.index = self.feature_summary.index.str.replace("dF/F0", self.amplitude_unit)
            self.peak_summary.columns = self.peak_summary.columns.str.replace("dF/F0", self.amplitude_unit)

        # reorganize the columns 
        feature_columns = self.feature_summary.index.sort_values()
        ordered_feature_columns = np.concatenate([['num beats', 'beat frequency [bpm]', 'RMSSD [s]', 'cross sectional area [mm2]', 'tissue area [mm2]', 'width [um]'],
                                                  feature_columns[feature_columns.str.contains('mean')],
                                                  ['SDRR [s]'],
                                                  feature_columns[feature_columns.str.contains('std')],
                                                  feature_columns[feature_columns.str.contains('min')],
                                                  feature_columns[feature_columns.str.contains('max')]])
        ordered_feature_columns = ordered_feature_columns[np.isin(ordered_feature_columns, feature_columns)]
        self.feature_summary = self.feature_summary[ordered_feature_columns]
        
        # export figures if toggled
        if savefig is True:
            self.save_trace(savefig_path)

    def save_trace(self, savefig_path):
        assert savefig_path is not None, "savefig_path cannot be None if savefig is True"

        os.makedirs(savefig_path, exist_ok=True)
        # First export the raw trace
        plt.plot(np.arange(len(self.raw_data))/self.frame_rate, self.raw_data, label="Raw data")
        if self.amplitude_unit:
            plt.plot(np.arange(len(self.baseline))/self.frame_rate, self.baseline, label="F₀ baseline")
        plt.scatter(self.peaks/self.frame_rate,self.raw_data[self.peaks], label = "Peaks")
        plt.scatter(self.valleys/self.frame_rate,self.raw_data[self.valleys], label = "Valleys")
        # plt.scatter(self.contraction_speeds_index/self.frame_rate, self.raw_data[self.contraction_speeds_index], label = "Contraction speed")
        # plt.scatter(self.relaxation_speeds_index/self.frame_rate, self.raw_data[self.relaxation_speeds_index], label = "Relaxation speed")
        plt.legend()
        plt.title(self.name+" raw_data")
        plt.xlabel("Time [s]")
        plt.ylabel("Raw amplitude [a.u.]")
        plt.savefig(os.path.join(savefig_path, self.name.replace("\\", "-")+" raw_data.png"))
        plt.clf()

        # export the velocity
        plt.plot(np.arange(len(self.velocity))/self.frame_rate, self.velocity, label="Velocity")
        plt.scatter(self.contraction_speeds_index/self.frame_rate, self.velocity[self.contraction_speeds_index], label = "Contraction speed")
        plt.scatter(self.relaxation_speeds_index/self.frame_rate, self.velocity[self.relaxation_speeds_index], label = "Relaxation speed")
        plt.legend()
        plt.title(self.name+" velocity")
        plt.xlabel("Time [s]")
        if self.amplitude_unit:
            plt.ylabel("Velocity [ΔF/F₀/s]")
        else:
            plt.ylabel("Velocity [a.u.]")
        plt.savefig(os.path.join(savefig_path, self.name.replace("\\", "-")+" velocity.png"))
        plt.clf()

        if self.amplitude_unit:
            # Then export the df_f0
            plt.plot(np.arange(len(self.df_f0))/self.frame_rate, self.df_f0)
            plt.scatter(self.peaks/self.frame_rate,self.df_f0[self.peaks], label = "Peaks")
            plt.scatter(self.valleys/self.frame_rate,self.df_f0[self.valleys], label = "Valleys")
            # plt.scatter(self.contraction_speeds_index/self.frame_rate, self.df_f0[self.contraction_speeds_index], label = "Contraction speed")
            # plt.scatter(self.relaxation_speeds_index/self.frame_rate, self.df_f0[self.relaxation_speeds_index], label = "Relaxation speed")
            plt.legend()
            plt.title(self.name+" ΔF/F₀")
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude [ΔF/F₀]")
            plt.savefig(os.path.join(savefig_path, self.name.replace("\\", "-")+" dF_F0.png"))
            plt.clf()

class TissueTrace(Trace):
    def __init__(self, data, frame_rate, max_bpm=360, name=None, skip_first_n_frames=0, width_px=None, area=None, resting_dist_px=None, um_per_pix=6.5, unloaded_dist=3000, force_disp_coeff=2.1):
        super().__init__(data, frame_rate, max_bpm, name, skip_first_n_frames)
        self.width_px = width_px
        self.um_per_pix = um_per_pix
        self.area = area
        self.resting_dist_px = resting_dist_px
        self.unloaded_dist = unloaded_dist
        self.force_disp_coeff = force_disp_coeff

    # override old method because calculate drift shouldn't exist
    def calculate_drift(self, method=None, min_prominence=None):
        pass
    def analyze_peaks(self, min_prominence=0.25, savefig=False, savefig_path=None):        
        # width in microns
        self.width = self.width_px * self.um_per_pix
        # cross sectional area in mm^2
        self.xc_area = np.pi * (self.width/1000/2)**2
        # pillar distance trace in microns
        self.distance = self.raw_data * self.um_per_pix
        # pillar deflection trace in microns
        self.df_f0 = self.unloaded_dist - self.distance
        # force trace in uN
        self.force = self.df_f0 * self.force_disp_coeff
        # resting_tension in uN. resting_tension = minimum passive tension for the whole video
        if self.resting_dist_px:
            self.resting_dist = self.resting_dist_px * self.um_per_pix
            self.resting_tension = (self.unloaded_dist - self.resting_dist) * self.force_disp_coeff
        else:
            self.resting_dist = self.distance.max() * self.um_per_pix
            self.resting_tension = self.force.min()
        # stress trace in mN/mm^2
        self.stress = self.force/1000/self.xc_area 
        
        # need to take care of some variables to make compatible with traditional trace analysis            
        self.amplitude_unit = "um" # boolean variable indicating whether df_f0 exists
        self.normalized = minmax_scale(self.df_f0)
        min_RR_index = 1 /(self.max_bpm / 60) * self.frame_rate 
        self.valleys, _ = sp.signal.find_peaks(-self.normalized, distance = max(2,min_RR_index), prominence=min_prominence, width=0, rel_height =1)
    
        # call parent method dont savefig until more params are calculated
        super().analyze_peaks(min_prominence, savefig=False, savefig_path=None)
        
        # drop any raw values because they are meaningless (raw amplitude)
        self.peak_summary = self.peak_summary[self.peak_summary.columns[~self.peak_summary.columns.str.contains('raw')]]
        self.feature_summary = self.feature_summary[self.feature_summary.index[~self.feature_summary.index.str.contains('raw')]]
        
        # rename amplitude to deflection
        self.peak_summary = self.peak_summary.rename(columns=dict(zip(self.peak_summary.columns, 
                                                  self.peak_summary.columns.str.replace("amplitude", "deflection"))))
        self.feature_summary = self.feature_summary.rename(index=dict(zip(self.feature_summary.index, 
                                                   self.feature_summary.index.str.replace("amplitude", "deflection"))))
        # calculate distance metrics
        self.peak_summary["active length [um]"] = self.distance[self.peaks]
        # passive distance for each peak is defined as (distance at the peak) - (deflection prominence at that peak)
        self.peak_summary["passive length [um]"] = self.distance[self.peaks] + sp.signal.peak_prominences(self.df_f0, self.peaks)[0]
        
        # calculate force metrics
        self.peak_summary["total force [uN]"] = (self.unloaded_dist - self.peak_summary["active length [um]"]) * self.force_disp_coeff
        self.peak_summary["passive tension [uN]"] = (self.unloaded_dist - self.peak_summary["passive length [um]"]) * self.force_disp_coeff
        self.peak_summary["active force [uN]"] = self.peak_summary["total force [uN]"] - self.resting_tension

        # calculate stress metrics
        self.peak_summary["total stress [mN/mm2]"] = self.peak_summary["total force [uN]"] / self.xc_area / 1000
        self.peak_summary["passive stress [mN/mm2]"] = self.peak_summary["passive tension [uN]"] / self.xc_area / 1000
        self.peak_summary["active stress [mN/mm2]"] = self.peak_summary["active force [uN]"] / self.xc_area / 1000
        
        # calculate work (W = 1/2(x_1^2-x_0^2)*force_disp_coeff)
        peak_deflections = self.unloaded_dist - self.peak_summary["active length [um]"]
        resting_deflections = self.unloaded_dist - self.peak_summary["passive length [um]"]
        self.peak_summary["work per contraction cycle [nJ]"] = 0.5 * self.force_disp_coeff * (peak_deflections ** 2 - resting_deflections ** 2) / 1000

        # add tissue measurements
        self.feature_summary["width [um]"] = self.width
        self.feature_summary["cross sectional area [mm2]"] = self.xc_area
        self.feature_summary["tissue area [mm2]"] = self.area * self.um_per_pix**2 # convert pix^2 to mm^2
        self.feature_summary["resting tension [uN]"] = self.resting_tension
        self.feature_summary["resting stress [mN/mm2]"] = self.resting_tension/1000/self.xc_area

        # calculate mean and std summary
        for col in ["active length [um]", "passive length [um]", 
                    "total force [uN]", "passive tension [uN]", "active force [uN]", 
                    "total stress [mN/mm2]", "passive stress [mN/mm2]", "active stress [mN/mm2]", 
                    "work per contraction cycle [nJ]"]:
            self.feature_summary['mean ' + col] = self.peak_summary[col].mean()
            self.feature_summary['std ' + col] = self.peak_summary[col].std()
            self.feature_summary['min ' + col] = self.peak_summary[col].min()
            self.feature_summary['max ' + col] = self.peak_summary[col].max()
        
        # reorganize the columns 
        feature_columns = self.feature_summary.index.sort_values()
        ordered_feature_columns = np.concatenate([['num beats', 'beat frequency [bpm]', 'RMSSD [s]', 'cross sectional area [mm2]', 'tissue area [mm2]', 'width [um]', "resting tension [uN]", 'resting stress [mN/mm2]'],
                                                  feature_columns[feature_columns.str.contains('mean')],
                                                  ['SDRR [s]'],
                                                  feature_columns[feature_columns.str.contains('std')],
                                                  feature_columns[feature_columns.str.contains('min')],
                                                  feature_columns[feature_columns.str.contains('max')]])
        ordered_feature_columns = ordered_feature_columns[np.isin(ordered_feature_columns, feature_columns)]
        self.feature_summary = self.feature_summary[ordered_feature_columns]

        # export figures if toggled
        if savefig is True:
            self.save_trace(savefig_path)
            
    # override save_trace 
    def save_trace(self, savefig_path):
        assert savefig_path is not None, "savefig_path cannot be None if savefig is True"

        os.makedirs(savefig_path, exist_ok=True)
        
        # Export the distance between the two anchors
        plt.plot(np.arange(len(self.distance))/self.frame_rate, self.distance, label="Distance")
        plt.axhline(y=self.resting_dist, label="Resting distance")
        plt.legend()
        plt.title(self.name+" anchor distance")
        plt.xlabel("Time [s]")
        plt.ylabel("Distance [μm]")
        plt.savefig(os.path.join(savefig_path, self.name.replace("\\", "-")+" distance.png"))
        plt.clf()
        
        # Export the deflection
        plt.plot(np.arange(len(self.df_f0))/self.frame_rate, self.df_f0, label="Deflection")
        plt.scatter(self.peaks/self.frame_rate,self.df_f0[self.peaks], label = "Max deflection")
        plt.scatter(self.valleys/self.frame_rate,self.df_f0[self.valleys], label = "Min deflection")
        # plt.scatter(self.contraction_speeds_index/self.frame_rate, self.df_f0[self.contraction_speeds_index], label = "Contraction speed")
        # plt.scatter(self.relaxation_speeds_index/self.frame_rate, self.df_f0[self.relaxation_speeds_index], label = "Relaxation speed")
        plt.axhline(y=self.unloaded_dist - self.resting_dist, label="Resting displacement")
        plt.legend()
        plt.title(self.name+" deflection")
        plt.xlabel("Time [s]")
        plt.ylabel("Deflection [μm]")
        plt.savefig(os.path.join(savefig_path, self.name.replace("\\", "-")+" deflection.png"))
        plt.clf()

        # Export the force
        plt.plot(np.arange(len(self.force))/self.frame_rate, self.force, label="Force")
        plt.scatter(self.peaks/self.frame_rate,self.force[self.peaks], label = "Max force")
        plt.scatter(self.valleys/self.frame_rate,self.force[self.valleys], label = "Passive tension")
        # plt.scatter(self.contraction_speeds_index/self.frame_rate, self.force[self.contraction_speeds_index], label = "Contraction speed")
        # plt.scatter(self.relaxation_speeds_index/self.frame_rate, self.force[self.relaxation_speeds_index], label = "Relaxation speed")
        plt.axhline(y=self.resting_tension, label="Resting tension")
        plt.legend()
        plt.title(self.name+" force")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [μN]")
        plt.savefig(os.path.join(savefig_path, self.name.replace("\\", "-")+" force.png"))
        plt.clf()
        
        # Export the stress
        plt.plot(np.arange(len(self.stress))/self.frame_rate, self.stress, label="Stress")
        plt.scatter(self.peaks/self.frame_rate,self.stress[self.peaks], label = "Max stress")
        plt.scatter(self.valleys/self.frame_rate,self.stress[self.valleys], label = "Passive stress")
        # plt.scatter(self.contraction_speeds_index/self.frame_rate, self.stress[self.contraction_speeds_index], label = "Contraction speed")
        # plt.scatter(self.relaxation_speeds_index/self.frame_rate, self.stress[self.relaxation_speeds_index], label = "Relaxation speed")
        plt.axhline(y=self.resting_tension/1000/self.xc_area, label="Resting stress")
        plt.legend()
        plt.title(self.name+" stress")
        plt.xlabel("Time [s]")
        plt.ylabel("Stress [mN/mm²]")
        plt.savefig(os.path.join(savefig_path, self.name.replace("\\", "-")+" stress.png"))
        plt.clf()
        
        # Export the velocity
        plt.plot(np.arange(len(self.velocity))/self.frame_rate, self.velocity, label="Velocity")
        plt.scatter(self.contraction_speeds_index/self.frame_rate, self.velocity[self.contraction_speeds_index], label = "Contraction speed")
        plt.scatter(self.relaxation_speeds_index/self.frame_rate, self.velocity[self.relaxation_speeds_index], label = "Relaxation speed")
        plt.legend()
        plt.title(self.name+" velocity")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [μm/s]")
        plt.savefig(os.path.join(savefig_path, self.name.replace("\\", "-")+" velocity.png"))
        plt.clf()

def get_config_param(dictionary, key):
    '''
    Helper function to extract parameter from the configuration file
    '''
    try:
        return dictionary[key]
    except:
        return {}

class BatchVideoAnalyzer():
    '''
    Batch video analyzer that takes folder path, file extension, and several analysis parameters 
    to execute analysis on multiple videos
    
    Attributes
    ----------
    folder_path : str or None
        path of the folder that will be recursively searched through for files that match file_ext
        e.g. "C:\\Users\\User\\Videos"
    file_ext : str or None
        file extension of video
        e.g. "mp4", "nd2", "tif"
    vid_type : str or None
        either "brightfield" or "fluorescent" indicating which video instance to create (BFVideo or FluoVideo)
    acq_mode : str or None
        videos acquired with pycromanager are organized in a specific way
        'pycromanager' indicates that the video was acquired with pycromanger
        None is everything else
    frame_rate : float or None
        frame rate of the video in fps
    config_file : str
        location of the configuation yaml file to be used to fill in all or some of the parameters

    Methods
    -------
    analyze_and_export(str reference) : None
        analyzes the videos into traces, then creates instance of BatchTraceAnalyzer
        to analyze the traces
    
    '''
    def __init__(self, folder_path=None, file_ext=None, vid_type=None, acq_mode=None, frame_rate=None, config_file=None, progressbar=None, progresstext=None):

        self.config_file = config_file
        self.progressbar = progressbar
        self.progresstext = progresstext

        # only relevant for the GUI application's progress bar
        # if progressbar exists, check if is an instance of QProgressBar
        # if not, set it as None as it is not updatable
        if self.progressbar:
            try:
                from PySide2.QtCore import Signal
                assert isinstance(self.progressbar, Signal)
            except:
                self.progressbar = None

        # same as above, but for progresstext
        if self.progresstext:
            try:
                from PySide2.QtCore import Signal
                assert isinstance(self.progresstext, Signal)
                self.progresstext.emit("Reading Videos")
            except:
                self.progresstext = None

        # load variables from the configuration yaml if it exists
        if self.config_file is None:
            self.init_config = {}
            self.analyze_and_export_config = {}
            self.video_config = {}
        else:
            with open(config_file, 'r') as config:
                self.config = yaml.full_load(config)
                self.video_config = get_config_param(self.config, 'Video')
                video_analyzer_config = get_config_param(self.config, 'BatchVideoAnalyzer')
                self.init_config = get_config_param(video_analyzer_config, '__init__')
                self.analyze_and_export_config = get_config_param(video_analyzer_config, 'analyze_and_export')
            if "folder_path" in self.init_config:
                self.folder_path = os.path.normpath(self.init_config["folder_path"])
            if "file_ext" in self.init_config:
                self.file_ext = self.init_config["file_ext"]
            if "vid_type" in self.init_config:
                self.vid_type = self.init_config["vid_type"]
            if "acq_mode" in self.init_config:
                self.acq_mode = self.init_config["acq_mode"]
            if "frame_rate" in self.init_config:
                self.frame_rate = self.init_config["frame_rate"]

        # if variables are explicitly specified, it should override the config file
        if folder_path is not None:
            self.folder_path = os.path.normpath(folder_path)
        if file_ext is not None:
            self.file_ext = file_ext
        if vid_type is not None:
            self.vid_type = vid_type
        if acq_mode is not None:
            self.acq_mode = acq_mode
        if frame_rate is not None:
            self.frame_rate = frame_rate

        # recursively search thru folder to find files of given extension
        self.file_list = glob.glob(os.path.normpath(os.path.join(self.folder_path, "**/*"+"."+self.file_ext)), recursive=True)

        if self.acq_mode == "pycromanager":
            self.file_list = np.unique(np.array([os.path.dirname(os.path.dirname(f)) for f in self.file_list]))

    def analyze_and_export(self, splits=None, split_units=None, skip_first_n=0, export_path=None, export_list=None):
        '''
        Calculates and exports desired data in a batch manner

        Parameters
        ----------
        export_path : str
            folder location of the export files to be saved
        export_list : list
            list of things to be exported. one or more of the following:
            ["summary", "peak_summary", "raw_trace", "df_f0", "plot_trace", "plot_mask", "beat_segments", "beat_segments_gaf"]

        Returns
        -------
        None
        '''
        # export_list: ["summary", "peak_summary", "raw_trace", "df_f0", "plot_trace", "plot_mask", "beat_segments", "beat_segments_gaf"]
        
        # load parameters from the configuration file
        # if they exist and are not explicitly overridden
        if not splits and 'splits' in self.analyze_and_export_config:
            splits = self.analyze_and_export_config['splits']
        if not split_units and 'split_units' in self.analyze_and_export_config:
            split_units = self.analyze_and_export_config['split_units']
        if not skip_first_n and 'skip_first_n' in self.analyze_and_export_config:
            skip_first_n = self.analyze_and_export_config['skip_first_n']
        if not export_path and 'export_path' in self.analyze_and_export_config:
            export_path = self.analyze_and_export_config['export_path']
        if not export_list and 'export_list' in self.analyze_and_export_config:
            export_list = self.analyze_and_export_config['export_list']

        if "plot_mask" in export_list:
            plot_mask = True
            plot_mask_path = os.path.join(export_path, "plot_mask")
        else:
            plot_mask = False
            plot_mask_path = None

        if "labeled_video" in export_list:
            labeled_video = True
            labeled_video_path = os.path.join(export_path, "labeled_video")
        else:
            labeled_video = False
            labeled_video_path = None

        if self.progressbar:
            self.progressbar.emit(0)

        self.raw_traces = pd.DataFrame()
        self.tissue_params = {}
        for i in range(len(self.file_list)):
            if self.file_list[i] == self.folder_path:
                name = os.path.basename(self.file_list[i])
            else:
                name = os.path.splitext(self.file_list[i].replace(self.folder_path, "")[1:])[0]

                # print("File path", self.file_list[i])
                # print("Folder path", self.folder_path)
                # print("Name", name)

            # update progress text if it is passed
            if self.progresstext:
                message = "Reading '" + name + "': " + str(i+1) + "/" + str(len(self.file_list)) + " videos."
                self.progresstext.emit(message)

#####################################################################################################################################################################################

            # try:
            #     if self.vid_type == "tissue":
            #         video = TissueVideo(self.file_list[i], acq_mode=self.acq_mode, frame_rate=self.frame_rate, name=name, **get_config_param(self.video_config, '__init__'))
            #         # for tissue video analysis, calculate mask and trace needs to be run first before width
            #         video.calculate_trace(savevid=labeled_video, savevid_path=labeled_video_path, **get_config_param(self.video_config, 'calculate_trace'))
            #         video.calculate_mask(savefig=plot_mask, savefig_path=plot_mask_path, **get_config_param(self.video_config, 'calculate_mask'))
            #         video.calculate_width(savefig=plot_mask, savefig_path=plot_mask_path)

            #         # save tissue specific parameters
            #         if not "width_pxs" in self.tissue_params.keys():
            #             self.tissue_params["width_pxs"] = []
            #         if not "tissue_areas" in self.tissue_params.keys():
            #             self.tissue_params["tissue_areas"] = []
            #         if not "resting_dist_pxs" in self.tissue_params.keys():
            #             self.tissue_params["resting_dist_pxs"] = []
            #         self.tissue_params["width_pxs"].append(video.width)
            #         self.tissue_params["tissue_areas"].append(video.area)
            #         self.tissue_params["resting_dist_pxs"].append(video.resting_dist)
            #     else:
            #         if self.vid_type == "fluorescent":
            #             video = FluoVideo(self.file_list[i], acq_mode=self.acq_mode, frame_rate=self.frame_rate, name=name, **get_config_param(self.video_config, '__init__'))
            #         else:
            #             video = BFVideo(self.file_list[i], acq_mode=self.acq_mode, frame_rate=self.frame_rate, name=name, **get_config_param(self.video_config, '__init__'))
            #         video.calculate_mask(savefig=plot_mask, savefig_path=plot_mask_path, **get_config_param(self.video_config, 'calculate_mask'))
            #         video.calculate_trace(**get_config_param(self.video_config, 'calculate_trace'))
            #     self.raw_traces = pd.concat([self.raw_traces, pd.Series(video.trace[1:], name = name).to_frame().T])
            # except:
            #     fail_message = "Skipping failed video '" + name + "': " + str(i+1) + "/" + str(len(self.file_list)) + " videos."
            #     print(fail_message)
            #     if self.progresstext:
            #         self.progresstext.emit(fail_message)

#####################################################################################################################################################################################

            if self.vid_type == "tissue":
                video = TissueVideo(self.file_list[i], acq_mode=self.acq_mode, frame_rate=self.frame_rate, name=name, **get_config_param(self.video_config, '__init__'))
                # for tissue video analysis, calculate mask and trace needs to be run first before width
                video.calculate_trace(savevid=labeled_video, savevid_path=labeled_video_path, **get_config_param(self.video_config, 'calculate_trace'))
                video.calculate_mask(savefig=plot_mask, savefig_path=plot_mask_path, **get_config_param(self.video_config, 'calculate_mask'))
                video.calculate_width(savefig=plot_mask, savefig_path=plot_mask_path)

                # save tissue specific parameters
                if not "width_pxs" in self.tissue_params.keys():
                    self.tissue_params["width_pxs"] = []
                if not "tissue_areas" in self.tissue_params.keys():
                    self.tissue_params["tissue_areas"] = []
                if not "resting_dist_pxs" in self.tissue_params.keys():
                    self.tissue_params["resting_dist_pxs"] = []
                self.tissue_params["width_pxs"].append(video.width)
                self.tissue_params["tissue_areas"].append(video.area)
                self.tissue_params["resting_dist_pxs"].append(video.resting_dist)
            else:
                if self.vid_type == "fluorescent":
                    video = FluoVideo(self.file_list[i], acq_mode=self.acq_mode, frame_rate=self.frame_rate, name=name, **get_config_param(self.video_config, '__init__'))
                else:
                    video = BFVideo(self.file_list[i], acq_mode=self.acq_mode, frame_rate=self.frame_rate, name=name, **get_config_param(self.video_config, '__init__'))
                video.calculate_mask(savefig=plot_mask, savefig_path=plot_mask_path, **get_config_param(self.video_config, 'calculate_mask'))
                video.calculate_trace(**get_config_param(self.video_config, 'calculate_trace'))
            self.raw_traces = pd.concat([self.raw_traces, pd.Series(video.trace[1:], name = name).to_frame().T])

#####################################################################################################################################################################################

            # update progressbar if it is passed. maximum is 50% because trace will be the 2nd half
            if self.progressbar:
                # percent of videos completed / 2
                new_value = int((i+1)/len(self.file_list)/2*100)
                self.progressbar.emit(new_value)

        # Pass the extracted traces to BatchTraceAnalyzer for the rest
        batch_trace_analyzer = BatchTraceAnalyzer(trace_type=self.vid_type, raw_traces=self.raw_traces, frame_rate=self.frame_rate, tissue_params=self.tissue_params, config_file=self.config_file, progressbar=self.progressbar, progressbar_from_video=True, progresstext=self.progresstext)
        batch_trace_analyzer.analyze_and_export(splits=splits, split_units=split_units,skip_first_n=skip_first_n, export_path=export_path, export_list=export_list)



class BatchTraceAnalyzer():
    '''
    Batch trace analyzer that takes either dataframe or csv data and several analysis parameters 
    to execute analysis on multiple traces
    
    Attributes
    ----------
    raw_traces : str or pd.DataFrame
        if type is str, location of the traces where the 1st column is the index and the rest are traces
        if type is pd DataFrame, each row is a trace
    frame_rate : float or None
        frame rate of the video in fps
    config_file : str
        location of the configuation yaml file to be used to fill in all or some of the parameters

    Methods
    -------
    analyze_and_export(str reference) : None
        analyzes the videos into traces, then creates instance of BatchTraceAnalyzer
        to analyze the traces
    
    '''
    def __init__(self, trace_type=None, raw_traces=None, frame_rate=None, tissue_params=None, config_file=None, progressbar=None, progressbar_from_video=False, progresstext=None):
        
        self.progressbar = progressbar
        self.progressbar_from_video = progressbar_from_video
        self.progresstext = progresstext

        # only relevant for the GUI application's progress bar
        # if progressbar exists, check if is an instance of QProgressBar
        # if not, set it as None as it is not updatable
        if self.progressbar:
            try:
                from PySide2.QtCore import Signal
                assert isinstance(self.progressbar, Signal)
            except:
                self.progressbar = None

        # same as above, but for progresstext
        if self.progresstext:
            try:
                from PySide2.QtCore import Signal
                assert isinstance(self.progresstext, Signal)
                self.progresstext.emit("Analyzing traces")
            except:
                self.progresstext = None

        self.config_file = config_file

        if self.config_file is None:
            self.trace_config = {}
        else:
            with open(config_file, 'r') as config:
                self.config = yaml.full_load(config)
                self.trace_config = get_config_param(self.config, 'Trace')
                trace_analyzer_config = get_config_param(self.config, 'BatchTraceAnalyzer')
                self.init_config = get_config_param(trace_analyzer_config, '__init__')
                self.analyze_and_export_config = get_config_param(trace_analyzer_config, 'analyze_and_export')
            if "trace_type" in self.init_config:
                self.trace_type = self.init_config["trace_type"]
            if "tissue_params" in self.init_config:
                self.tissue_params = self.init_config["tissue_params"]
            if "frame_rate" in self.init_config:
                self.frame_rate = self.init_config["frame_rate"]
            if "raw_traces" in self.init_config:
                self.raw_traces = self.init_config["raw_traces"]

        # if explicitly specified, override the config file
        if trace_type is not None:
            self.trace_type = trace_type
        if tissue_params is not None:
            self.tissue_params = tissue_params
        if frame_rate is not None:
            self.frame_rate = frame_rate
        if raw_traces is not None:
            self.raw_traces = raw_traces

        # ensure that raw_traces are a path to csv or a pd DataFrame
        if type(self.raw_traces) == str:
            self.raw_traces_path = self.raw_traces
            self.raw_traces = pd.read_csv(self.raw_traces_path, index_col=0)
        elif type(self.raw_traces) == pd.core.frame.DataFrame:
            self.raw_traces = raw_traces
        else:
            raise TypeError("'raw_traces' parameter must be a file path or a pandas Dataframe.")

    def analyze_and_export(self, splits=None, split_units=None, skip_first_n=0, export_path=None, export_list=None):
        '''
        Calculates and exports desired data in a batch manner

        Parameters
        ----------
        export_path : str
            folder location of the export files to be saved
        export_list : list
            list of things to be exported. one or more of the following:
            ["summary", "peak_summary", "raw_trace", "df_f0", "plot_trace", "plot_mask", "beat_segments", "beat_segments_gaf"]

        Returns
        -------
        None
        '''
        # load parameters from the configuration file
        # if they exist and are not explicitly overridden
        if not splits and 'splits' in self.analyze_and_export_config:
            splits = self.analyze_and_export_config['splits']
        if not split_units and 'split_units' in self.analyze_and_export_config:
            split_units = self.analyze_and_export_config['split_units']
        if not skip_first_n and 'skip_first_n' in self.analyze_and_export_config:
            skip_first_n = self.analyze_and_export_config['skip_first_n']
        if not export_path and 'export_path' in self.analyze_and_export_config:
            export_path = self.analyze_and_export_config['export_path']
        if not export_list and 'export_list' in self.analyze_and_export_config:
            export_list = self.analyze_and_export_config['export_list']

        os.makedirs(export_path, exist_ok=True)
    
        if "plot_trace" in export_list:
            plot_trace = True
            plot_trace_path = os.path.join(export_path, "plot_trace")
        else:
            plot_trace = False
            plot_trace_path = None

        # if no splits, set 1 split with the range as the full trace
        if not splits:
            splits = [[0,None]]

        # convert splits into units of frames (split_units are either in 'seconds' or 'frames')
        if split_units == "seconds":
            # multiply by self.frame_rate if it's not None
            splits_frames = [[int(np.round(sec * self.frame_rate)) if sec else None for sec in split] for split in splits]
            skip_first_n_frames = int(np.round(skip_first_n * self.frame_rate))
        else:
            splits_frames = splits
            skip_first_n_frames = skip_first_n

        total_summary_splits = []
        peak_summary_splits = []
        raw_data_splits = []
        df_f0_splits = []
        velocity_splits = []
        segments_splits = []
        distance_splits = []
        force_splits = []
        stress_splits = []
        

        for split_i in range(len(splits)):
            split = splits[split_i]
            total_summary = pd.DataFrame()
            peak_summary = pd.DataFrame()
            df_f0 = pd.DataFrame()
            velocity = pd.DataFrame()
            segments = pd.DataFrame()
            if self.trace_type == "tissue":
                distance = pd.DataFrame()
                force = pd.DataFrame()
                stress = pd.DataFrame()
            else:
                raw_data = pd.DataFrame()

            for i in range(len(self.raw_traces)):

                # update progress text if it is passed
                if self.progresstext:
                    trace_num = split_i * len(self.raw_traces) + i + 1
                    message = "Analyzing trace '" + self.raw_traces.index[i] + " split " + str(split_i+1) + "': " + str(trace_num) + "/" + str(len(splits) * len(self.raw_traces)) + " traces."
                    self.progresstext.emit(message)

#####################################################################################################################################################################################

                # try: 
                #     # batch analysis of videos of different sizes results in nan values at the end
                #     data_numpy = self.raw_traces.iloc[i, splits_frames[split_i][0]:splits_frames[split_i][1]].to_numpy()
                #     data_numpy = data_numpy[~np.isnan(data_numpy)]
                #     if len(splits) == 1:
                #         name = self.raw_traces.index[i]
                #     else:
                #         name = self.raw_traces.index[i] + " split " + str(split_i+1)
                    
                #     # suppress warnings
                #     # warnings are thrown when baseline fit doesnt converge well or if the number of peaks is < 3
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("ignore")
                #         if self.trace_type == "tissue":
                #             trace = TissueTrace(data=data_numpy, name=name, frame_rate=self.frame_rate, 
                #                 width_px=self.tissue_params["width_pxs"][i], area=self.tissue_params["tissue_areas"][i], 
                #                 resting_dist_px=self.tissue_params["resting_dist_pxs"][i], skip_first_n_frames=skip_first_n_frames,
                #                 **get_config_param(self.trace_config, '__init__'))
                #         else:
                #             trace = Trace(data=data_numpy, name=name, frame_rate=self.frame_rate, skip_first_n_frames=skip_first_n_frames, **get_config_param(self.trace_config, '__init__'))
                #             trace.calculate_drift(**get_config_param(self.trace_config, 'calculate_drift'))
                #         trace.analyze_peaks(savefig=plot_trace, savefig_path=plot_trace_path, **get_config_param(self.trace_config, 'analyze_peaks'))
                #     total_summary = pd.concat([total_summary,trace.feature_summary.to_frame().T])
                #     peak_summary = pd.concat([peak_summary, trace.peak_summary])
                #     df_f0 = pd.concat([df_f0, pd.Series(trace.df_f0, name=name).to_frame().T])
                #     velocity = pd.concat([velocity, pd.Series(trace.velocity, name=name).to_frame().T])
                #     segments = pd.concat([segments, trace.beat_segments])
                #     if self.trace_type == "tissue":
                #         distance = pd.concat([distance, pd.Series(trace.distance, name=name).to_frame().T])
                #         force = pd.concat([force, pd.Series(trace.force, name=name).to_frame().T])
                #         stress = pd.concat([stress, pd.Series(trace.stress, name=name).to_frame().T])
                #     else:
                #         raw_data = pd.concat([raw_data, pd.Series(trace.raw_data, name=name).to_frame().T])

                # except:
                #     fail_message = "Skipping failed trace analysis: '" + self.raw_traces.index[i] + " split " + str(split_i+1) + "': " + str(trace_num) + "/" + str(len(splits) * len(self.raw_traces)) + " traces."
                #     print(fail_message)
                #     if self.progresstext:
                #         self.progresstext.emit(fail_message)

#####################################################################################################################################################################################

                # batch analysis of videos of different sizes results in nan values at the end
                data_numpy = self.raw_traces.iloc[i, splits_frames[split_i][0]:splits_frames[split_i][1]].to_numpy()
                data_numpy = data_numpy[~np.isnan(data_numpy)]
                if len(splits) == 1:
                    name = self.raw_traces.index[i]
                else:
                    name = self.raw_traces.index[i] + " split " + str(split_i+1)
                
                # suppress warnings
                # warnings are thrown when baseline fit doesnt converge well or if the number of peaks is < 3
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if self.trace_type == "tissue":
                        trace = TissueTrace(data=data_numpy, name=name, frame_rate=self.frame_rate, 
                            width_px=self.tissue_params["width_pxs"][i], area=self.tissue_params["tissue_areas"][i], 
                            resting_dist_px=self.tissue_params["resting_dist_pxs"][i], skip_first_n_frames=skip_first_n_frames,
                            **get_config_param(self.trace_config, '__init__'))
                    else:
                        trace = Trace(data=data_numpy, name=name, frame_rate=self.frame_rate, skip_first_n_frames=skip_first_n_frames, **get_config_param(self.trace_config, '__init__'))
                        trace.calculate_drift(**get_config_param(self.trace_config, 'calculate_drift'))
                    trace.analyze_peaks(savefig=plot_trace, savefig_path=plot_trace_path, **get_config_param(self.trace_config, 'analyze_peaks'))
                total_summary = pd.concat([total_summary,trace.feature_summary.to_frame().T])
                peak_summary = pd.concat([peak_summary, trace.peak_summary])
                df_f0 = pd.concat([df_f0, pd.Series(trace.df_f0, name=name).to_frame().T])
                velocity = pd.concat([velocity, pd.Series(trace.velocity, name=name).to_frame().T])
                segments = pd.concat([segments, trace.beat_segments])
                if self.trace_type == "tissue":
                    distance = pd.concat([distance, pd.Series(trace.distance, name=name).to_frame().T])
                    force = pd.concat([force, pd.Series(trace.force, name=name).to_frame().T])
                    stress = pd.concat([stress, pd.Series(trace.stress, name=name).to_frame().T])
                else:
                    raw_data = pd.concat([raw_data, pd.Series(trace.raw_data, name=name).to_frame().T])

#####################################################################################################################################################################################

                # update progressbar if it is passed. if passed from video, start from 50%
                if self.progressbar:
                    trace_num = split_i * len(self.raw_traces) + i + 1
                    # percent of traces completed / 2 + 50%
                    if self.progressbar_from_video:
                        new_value = int(trace_num/(len(splits) * len(self.raw_traces))/2*100 + 50)
                    # percent of traces completed
                    else:
                        new_value = int(trace_num/(len(splits) * len(self.raw_traces))*100)
                    self.progressbar.emit(min(new_value, 100))  

            # drop empty columns (i.e. velocity with vs without units, df/f0 amplitude, etc)
            total_summary = total_summary.dropna(how="all", axis=1)
            peak_summary = peak_summary.dropna(how="all", axis=1)

            total_summary_splits.append(total_summary)
            peak_summary_splits.append(peak_summary)

            # relabel the columns in units of seconds
            timestamp = np.linspace(start=0, stop=df_f0.shape[1]/self.frame_rate, num=df_f0.shape[1])
            df_f0 = df_f0.rename(columns=dict(zip(df_f0.columns, timestamp)))
            velocity = velocity.rename(columns=dict(zip(velocity.columns, timestamp)))

            seg_timestamp = np.linspace(start=0, stop=segments.shape[1]/self.frame_rate, num=segments.shape[1])
            segments = segments.rename(columns=dict(zip(segments.columns, seg_timestamp)))

            df_f0_splits.append(df_f0)
            velocity_splits.append(velocity)
            segments_splits.append(segments)

            if self.trace_type == "tissue":
                distance = distance.rename(columns=dict(zip(distance.columns, timestamp)))
                force = force.rename(columns=dict(zip(force.columns, timestamp)))
                stress = stress.rename(columns=dict(zip(stress.columns, timestamp)))

                distance_splits.append(distance)
                force_splits.append(force)
                stress_splits.append(stress)
            else:
                raw_data = raw_data.rename(columns=dict(zip(raw_data.columns, timestamp)))
                raw_data_splits.append(raw_data)

        if "summary" in export_list:
            summary_writer =  pd.ExcelWriter(os.path.join(export_path, "summary.xlsx"))
        if "peak_summary" in export_list:
            peak_summary_writer = pd.ExcelWriter(os.path.join(export_path, "peak_summary.xlsx"))
        if "traces" in export_list:
            velocity_writer = pd.ExcelWriter(os.path.join(export_path, "velocity.xlsx"))
            if self.trace_type == "tissue":
                distance_writer = pd.ExcelWriter(os.path.join(export_path, "pillar_distance.xlsx"))
                df_f0_writer = pd.ExcelWriter(os.path.join(export_path, "pillar_deflection.xlsx"))
                force_writer = pd.ExcelWriter(os.path.join(export_path, "force.xlsx"))
                stress_writer = pd.ExcelWriter(os.path.join(export_path, "stress.xlsx"))
            else:
                raw_trace_writer = pd.ExcelWriter(os.path.join(export_path, "raw_trace.xlsx"))
                df_f0_writer = pd.ExcelWriter(os.path.join(export_path, "df_f0.xlsx"))

        if "beat_segments" in export_list:
            beat_segments_writer = pd.ExcelWriter(os.path.join(export_path, "beat_segments.xlsx")) 
    
        for split_i in range(len(splits)):
            if len(splits) == 1:
                split_name = "full_trace"
            elif split_i == len(splits)-1:
                split_name = "split_" + str(split_i+1) + " (" + str(splits[split_i][0]) + "-end) " + split_units
            else:
                split_name = "split_" + str(split_i+1) + " (" + str(splits[split_i][0]) + "-" + str(splits[split_i][1]) + ") " + split_units

            if "summary" in export_list:
                total_summary_splits[split_i].to_excel(summary_writer, sheet_name=split_name, engine='xlsxwriter')
            if "peak_summary" in export_list:
                peak_summary_splits[split_i].to_excel(peak_summary_writer, sheet_name=split_name, engine='xlsxwriter')
            if "traces" in export_list:
                df_f0_splits[split_i].to_excel(df_f0_writer, sheet_name=split_name, engine='xlsxwriter')
                velocity_splits[split_i].to_excel(velocity_writer, sheet_name=split_name, engine='xlsxwriter')
                if self.trace_type == "tissue":
                    distance_splits[split_i].to_excel(distance_writer, sheet_name=split_name, engine='xlsxwriter')
                    force_splits[split_i].to_excel(force_writer, sheet_name=split_name, engine='xlsxwriter')
                    stress_splits[split_i].to_excel(stress_writer, sheet_name=split_name, engine='xlsxwriter')
                else:
                    raw_data_splits[split_i].to_excel(raw_trace_writer, sheet_name=split_name, engine='xlsxwriter')
            if "beat_segments" in export_list:
                segments_splits[split_i].to_excel(beat_segments_writer, sheet_name=split_name, engine='xlsxwriter')

            if "beat_segments_gaf" in export_list:
                import pyts.image
                # truncate beats to 2 seconds and pad nans with the last sample
                segment_len = int(2 * self.frame_rate)
                def process_segments(segments):
                    processed_seg = segments.copy()
                    # pad df if it's shorter than 2 seconds
                    if processed_seg.shape[1] < segment_len:
                        processed_seg = pd.concat([processed_seg, pd.DataFrame(columns=range(processed_seg.shape[1], segment_len))],axis=1)
                    def peak_truncater2(index_num):
                        data = processed_seg.iloc[index_num, 0:].to_numpy()
                        argmax_derivative = np.nanargmax(np.diff(data)) #nanargmax instead of argmax to ignore the nans 
                        start_i = max(0, argmax_derivative-self.frame_rate/10) # start 0.1 sec before the peak derivative
                        end_i = min(start_i + segment_len - 1, len(data))
                        return processed_seg.iloc[index_num, int(start_i):int(end_i)].dropna()
                    
                    for i in range(len(processed_seg)):
                        truncated = peak_truncater2(i).to_numpy()
                        processed_seg.iloc[i, 0: len(truncated)] = truncated
                        processed_seg.iloc[i, len(truncated):] = truncated[-1]
                    return processed_seg.iloc[:,:segment_len]
                processed_seg = process_segments(segments_splits[split_i])

                if len(processed_seg) > 0:
                    gaf = pyts.image.GramianAngularField()
                    segments_gaf = gaf.fit_transform(processed_seg)
                    os.makedirs(os.path.join(export_path, "gaf"), exist_ok=True)
                    for i in range(len(segments_gaf)):
                        file_name = str(segments_splits[split_i].index[i]) + "_" + split_name + ".png"
                        plt.imsave(os.path.join(export_path, "gaf", file_name), segments_gaf[i], cmap="gray")

        if 'summary_writer' in locals():
            summary_writer.close()
        if 'peak_summary_writer' in locals():
            peak_summary_writer.close()
        if 'raw_trace_writer' in locals():
            raw_trace_writer.close()
        if 'df_f0_writer' in locals():
            df_f0_writer.close()
        if 'distance_writer' in locals():
            distance_writer.close()
        if 'force_writer' in locals():
            force_writer.close()
        if 'stress_writer' in locals():
            stress_writer.close()
        if 'velocity_writer' in locals():
            velocity_writer.close()
        if 'beat_segments_writer' in locals():
            beat_segments_writer.close()

        if self.progressbar:
            self.progressbar.emit(100)
        if self.progresstext:
            self.progresstext.emit("Analysis complete. Check folder for exported files.")
