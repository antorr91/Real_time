import sounddevice as sd 
import numpy as np
import madmom
import librosa as lb
import csv
import os
import pandas as pd
import warnings
from features_realtime import (spectral_centroid_realtime, 
    rms_features_realtime,
    compute_envelope_features_realtime,
    compute_f0_features_realtime, get_calls_waveform, bp_filter
)
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
import datetime
# Audio parameters
duration = 30  # Duration of recording in seconds
sample_rate = 44100  # Sampling frequency
buffer_size = 2048  # Buffer size for efficient analysis
hop_length = 512 # Hop length from your offline study

# Parameters specific to chick vocalizations (from offline data)
MIN_DURATION = 0.02  # Minimum duration of a vocalization (from your data)
MAX_DURATION = 0.54  # Maximum duration (from your data)
AV_DURATION = 0.19   # Average duration (from your data)




# Spectral parameters for chicks (from offline data)
SPEC_NUM_BANDS = 15
SPEC_FMIN = 2500
SPEC_FMAX = 5000
SPEC_FREF = 2800
PP_THRESHOLD = 1.8
PP_PRE_AVG = 25
PP_POST_AVG = 1
PP_PRE_MAX = 3
PP_POST_MAX = 2


# Parameters for the onset detection functions
# HFC_parameters = {'hop_length': 441, 'sr':44100, 'spec_num_bands':15, 'spec_fmin': 2500, 'spec_fmax': 5000, 'spec_fref': 2800,
                #   'pp_threshold':  1.8, 'pp_pre_avg':25, 'pp_post_avg':1, 'pp_pre_max':3, 'pp_post_max':2,'global shift': 0.1, 'double_onset_correction': 0.1}

# Parameters for offset detection
OFFSET_FMIN = 2050
OFFSET_FMAX = 8000
OFFSET_N_MELS = 15


# def offset_detection_first_order(file_name, onsets,  min_duration= MIN_DURATION, max_duration= MAX_DURATION, av_duration= AV_DURATION):

#     spectrogram= lb.feature.melspectrogram(y=y, sr=44100, hop_length=512, n_fft=2048 * 2, window=0.12, fmin= 2050, fmax=8000, n_mels= 15)
    


# Output CSV file
csv_filename = os.path.join("C:\\Users\\anton\\Real_time", "chick_calls_detection.csv")

save_directory= "C:\\Users\\anton\\Real_time\\features"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Global variables to maintain state between blocks
global_time = 0
audio_buffer = np.array([])
last_processed_time = 0
detection_buffer = []  # Buffer to avoid duplicates

# Write CSV headers
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Onset (s)", "Offset (s)", "Duration (s)", "Call Number", 
                     "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis",
                     "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope",
                     "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", 
                     "F2-F0 Ratio", "Spectral Centroid Mean",
                     "Spectral Centroid Std", "RMS Mean", "RMS Std",
                     "Slope", "Attack_magnitude", "Attack_time"])

def detect_onset_offset(audio_data, sr, global_time_offset):
    """Detects onsets and offsets in audio data in real-time using optimized parameters."""
    
    # Calculate spectrogram filtered exactly as in your offline study
    spec_mdm = madmom.audio.spectrogram.FilteredSpectrogram(
        audio_data, 
        sample_rate=sr, 
        num_bands=SPEC_NUM_BANDS, 
        fmin=SPEC_FMIN, 
        fmax=SPEC_FMAX, 
        fref=SPEC_FREF,
        norm_filters=True, 
        unique_filters=True
    )
    
    # Onset detection with parameters from your study
    activation = madmom.features.onsets.high_frequency_content(spec_mdm)
    peaks = madmom.features.onsets.peak_picking(
        activation, 
        threshold=PP_THRESHOLD, 
        smooth=None,
        pre_avg=PP_PRE_AVG, 
        post_avg=PP_POST_AVG, 
        pre_max=PP_PRE_MAX, 
        post_max=PP_POST_MAX
    )
    
    # Convert frame indices to time (seconds) and add global time offset
    onsets_seconds = np.array([p * hop_length / sr + global_time_offset for p in peaks])
    
    if len(onsets_seconds) == 0:
        return np.array([]), np.array([])
    
    # Spectrogram for offset detection (as in your offline study)
    spectrogram = lb.feature.melspectrogram(
        y=audio_data, 
        sr=sr, 
        hop_length=512, 
        n_fft=2048 * 2, 
        window='hann',
        fmin=OFFSET_FMIN, 
        fmax=OFFSET_FMAX, 
        n_mels=OFFSET_N_MELS
    )
    
    # Calculate times relative within the current buffer
    local_onsets = np.array([onset - global_time_offset for onset in onsets_seconds])
    
    # Calculate windows for offset search (as in your offline study)
    min_duration_from_onsets = local_onsets + MIN_DURATION
    max_duration_from_onsets = local_onsets + MAX_DURATION
    
    # Convert to frames
    min_frames = lb.time_to_frames(min_duration_from_onsets, sr=sr, hop_length=512)
    max_frames = lb.time_to_frames(max_duration_from_onsets, sr=sr, hop_length=512)
    
    # Ensure indices are valid
    max_frame = spectrogram.shape[1] - 1
    min_frames = np.clip(min_frames, 0, max_frame)
    max_frames = np.clip(max_frames, 0, max_frame)
    
    offsets = []
    valid_onsets = []
    
    for i, (startw, onset_time) in enumerate(zip(min_frames, onsets_seconds)):
        if i < len(max_frames):
            endw = max_frames[i]
            
            # Check overlaps with other onsets (as in your offline algorithm)
            onsets_frames = lb.time_to_frames(local_onsets, sr=sr, hop_length=512)
            for onset in onsets_frames:
                if onset > startw and onset < endw:
                    endw = onset - 1
                    break
            
            # Check validity of window
            if startw >= endw or startw >= spectrogram.shape[1] or endw >= spectrogram.shape[1]:
                continue
            
            # Extract spectrogram window
            spectrogram_window = spectrogram[:, startw:endw]
            if spectrogram_window.size == 0:
                continue
            
            # Calculate mean across frequency bands
            average_spectrogram = np.mean(spectrogram_window, axis=0)
            if average_spectrogram.size < 2:
                continue
            
            # Calculate first-order difference (as in your offline algorithm)
            y_diff = np.diff(average_spectrogram, n=1)
            if y_diff.size == 0:
                continue
            
            # Find minimum point (as in your offline algorithm)
            n_min = np.argmin(y_diff)
            offset_in_frames = startw + n_min
            offset_time = lb.frames_to_time(offset_in_frames, sr=sr, hop_length=512) + global_time_offset
            
            # Ensure offset is after onset
            if offset_time > onset_time:
                offsets.append(offset_time)
                valid_onsets.append(onset_time)
    
    return np.array(valid_onsets), np.array(offsets)

def is_duplicate_event(onset, detection_buffer, time_threshold=0.020):
    """Checks if an event is a duplicate of a previously detected one."""
    for past_onset, _ in detection_buffer:
        if abs(onset - past_onset) < time_threshold:
            return True
    return False



def global_shift_correction(predicted_onsets, shift):
    '''subtract shift second to all the predicted onsets.
    Args:
        predicted_onsets (list): List of predicted onsets.
        shift (float): Global shift in seconds.
    Returns:
        list: Corrected predicted onsets.
    '''
    # compute global shift
    corrected_predicted_onsets = []
    for po in predicted_onsets:
        #subtract a global shift of 0.01 ms or more  to all the predicted onsets
        if po - shift > 0: # to avoid negative onsets
            corrected_predicted_onsets.append(po - shift)
        else:
            continue

    return np.array(corrected_predicted_onsets)




def double_onset_correction(onsets_predicted, correction= 0.020):
    '''Correct double onsets by removing onsets which are less than a given threshold in time.
    Args:
        onsets_predicted (list): List of predicted onsets.
        gt_onsets (list): List of ground truth onsets.
        correction (float): Threshold in seconds.
    Returns:
        list: Corrected predicted onsets.
    '''    
    # Calculate interonsets difference
    #gt_onsets = np.array(gt_onsets, dtype=float)

    # Calculate the difference between consecutive onsets
    differences = np.diff(onsets_predicted)

    # Create a list to add the filtered onset and add a first value

    filtered_onsets = [onsets_predicted[0]]  #Add the first onset

    # Subtract all the onsets which are less than fixed threshold in time
    for i, diff in enumerate(differences):
      if diff >= correction:
      # keep the onset if the difference is more than the given selected time
        filtered_onsets.append(onsets_predicted[i + 1])
        #print the number of onsets predicted after correction
    return np.array(filtered_onsets)














# def process_audio(indata, frames, time, status):
#     global audio_buffer, global_time, last_processed_time, detection_buffer
    
#     if status:
#         print(f"Status: {status}")
    
#     # Add new audio block to buffer
#     current_audio = np.squeeze(indata)
#     audio_buffer = np.concatenate((audio_buffer, current_audio))
    
#     # Update global time
#     block_duration = frames / sample_rate
#     global_time += block_duration
    
#     # Process only if we have enough data
#     min_buffer_size = 1.5 * sample_rate  # Minimum size to have enough data for analysis
#     if len(audio_buffer) >= min_buffer_size:
#         # Calculate start timestamp for this block
#         start_time = global_time - (len(audio_buffer) / sample_rate)
        
#         # Detect onset and offset
#         onsets_seconds, offsets_seconds = detect_onset_offset(audio_buffer, sample_rate, start_time)
              
#         # Filter out events that have already been detected
#         new_events = []
#         for onset, offset in zip(onsets_seconds, offsets_seconds):
#             if onset > last_processed_time and not is_duplicate_event(onset, detection_buffer):
#                 duration = offset - onset
#                 # Check if duration is within expected limits
#                 if MIN_DURATION <= duration <= MAX_DURATION:
#                     new_events.append((onset, offset, duration))
#                     # Add to deduplication buffer
#                     detection_buffer.append((onset, offset))
        
#         # Limit size of deduplication buffer
#         while len(detection_buffer) > 20:
#             detection_buffer.pop(0)
            
#         # Update last processed time
#         if len(onsets_seconds) > 0:
#             last_processed_time = max(last_processed_time, max(onsets_seconds))
        
#         if new_events:
#             print(f"Detected {len(new_events)} new call:")
#             for onset, offset, duration in new_events:
#                 print(f"  Onset: {onset:.3f}s, Offset: {offset:.3f}s, Duration: {duration:.3f}s")

#         # Save results to CSV
#             with open(csv_filename, mode='a', newline='') as file:
#                 writer = csv.writer(file)
#                 for onset, offset, duration in new_events:
#                     writer.writerow([f"{onset:.6f}", f"{offset:.6f}", f"{duration:.6f}"])
        
#         # Keep only the last second of audio for overlap
#         overlap_samples = int(1 * sample_rate)
#         if len(audio_buffer) > overlap_samples:
#             audio_buffer = audio_buffer[-overlap_samples:]




