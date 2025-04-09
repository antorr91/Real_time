import numpy as np
import os
import pandas as pd
import librosa as lb
import matplotlib.pyplot as plt
# import utils as ut
import scipy.signal as signal
from scipy.signal import hilbert
import scipy.stats as stats
import soundfile as sf


# offline parameters
# frame_length = 2048
# hop_length = 512
# win_length = frame_length // 2
# n_fft = 2048*2

def spectral_centroid_realtime(mel_spectrogram, onsets, offsets, sr, frame_length, hop_length):
    """Compute spectral centroid features for each call (realtime) using mel-spectrogram."""
    sc_mean = []
    sc_std = []


    # mel_spec = lb.feature.melspectrogram(y=audio_fy, sr=sr, n_fft=frame_length, 
    #                                     hop_length=hop_length, n_mels=128, fmin=2000, fmax=12600)

        
    # # Extract the spectrogram calls from the audio file
    # calls_s_files = segment_spectrogram(spectrogram= mel_spec, onsets=onsets_sec, offsets=offsets_sec, sr=sr)
      

    onset_frames = lb.time_to_frames(onsets, sr=sr, hop_length=hop_length)
    offset_frames = lb.time_to_frames(offsets, sr=sr, hop_length=hop_length)

    for start, end in zip(onset_frames, offset_frames):
        call_spec = mel_spectrogram[:, start:end]
        spectral_centroid_call = lb.feature.spectral_centroid(S=call_spec, sr=sr, n_fft=frame_length, hop_length=hop_length)

        spectral_centroid_call_without_nans = spectral_centroid_call[~np.isnan(spectral_centroid_call)]
        if len(spectral_centroid_call_without_nans) == 0:
            sc_mean.append(np.nan)
            sc_std.append(np.nan)
            continue 
        else:     

            sc_mean.append(np.mean(spectral_centroid_call_without_nans))
            sc_std.append(np.std(spectral_centroid_call_without_nans))

    return sc_mean, sc_std




def rms_features_realtime(calls_waveform, frame_length, hop_length):
    """Compute RMS mean and std for each call waveform segment."""
    rms_mean = []
    rms_std = []

    for call in calls_waveform:
        rms = lb.feature.rms(y=call, frame_length=frame_length, hop_length=hop_length)

        rms_call_without_nans = rms[~np.isnan(rms)]
        # check if the call is empty
        if len(rms_call_without_nans) == 0:
            rms_mean.append(np.nan)
            rms_std.append(np.nan)
            
        else:
            rms_mean.append(np.mean(rms_call_without_nans))
            rms_std.append(np.std(rms_call_without_nans))

    return rms_mean, rms_std



def compute_envelope_features_realtime(calls_waveform, sr=44100):
    ''' Compute the envelope of each call and extract relevant features. '''

    Slopes = []
    Attack_magnitude = []
    Attack_time = []

    for call in calls_waveform:
        # Compute the analytic signal
        analytic_signal = hilbert(call)
        # Compute the envelope
        envelope = np.abs(analytic_signal)
        # Compute the onset index (assumiamo che l'onset sia all'inizio del segnale)
        onset_in_time = 0  # o utilizza lb.time_to_frames se specifico per l'onset
        onset= lb.frames_to_time(onset_in_time, sr=sr)  # Converti in secondi

        # Compute the peak index
        peak_index = np.argmax(envelope)


        peak_index_in_time = lb.frames_to_time(peak_index, sr=sr)
        attack_magnitude = envelope[peak_index] - envelope[onset]

        attack_time = peak_index_in_time - onset_in_time

        # Compute the slope of the envelope
        slope = attack_magnitude / attack_time 
        # slope = attack_magnitude / attack_time if attack_time > 0 else 0
        # if np.isnan(slope) or slope < 0:
        #     slope = 0
        Slopes.append(slope)
        Attack_magnitude.append(attack_magnitude)
        Attack_time.append(attack_time)

    # Crea un DataFrame per i risultati
    envelope_features_calls = pd.DataFrame({
        'Slope': Slopes,
        'Attack_magnitude': Attack_magnitude,
        'Attack_time': Attack_time
    })

    return envelope_features_calls



# def compute_f0_features_realtime(audio, onsets, offsets, sr, hop_length, frame_length, n_fft, pyin_fmin_hz, pyin_fmax_hz, pyin_beta, pyin_ths, pyin_resolution):
#     ''' Compute F0 features for each call using PYIN. '''
#     f0_means = []
#     f0_stds = []
#     f0_skewnesses = []
#     f0_kurtosises = []
#     f0_bandwidths = []
#     f0_fst_order_diffs_means = []
#     f0_slopes = []
#     f0_mag_means = []
#     f1_mag_means = []
#     f2_mag_means = []
#     f1_f0_ratios_mag_means = []
#     f2_f0_ratios_mag_means = []

#     # Estimare la frequenza fondamentale usando PYIN
#     f0_pyin_lb, _, _ = lb.pyin(audio, sr=sr, frame_length=frame_length, hop_length=hop_length, 
#                                fmin=pyin_fmin_hz, fmax=pyin_fmax_hz, n_thresholds=pyin_ths, 
#                                beta_parameters=pyin_beta, resolution=pyin_resolution)

    
    
    
    

#     # Segmenta F0 in base agli onset e offset
#     for start, end in zip(onsets, offsets):
#         f0_call = f0_pyin_lb[int(start * sr / hop_length):int(end * sr / hop_length)]
#         f0_call = f0_call[~np.isnan(f0_call)]  # Rimuovi NaN

#         if len(f0_call) == 0:
#             f0_means.append(np.nan)
#             f0_stds.append(np.nan)
#             f0_skewnesses.append(np.nan)
#             f0_kurtosises.append(np.nan)
#             f0_bandwidths.append(np.nan)
#             f0_fst_order_diffs_means.append(np.nan)
#             f0_slopes.append(np.nan)
#             f0_mag_means.append(np.nan)
#             f1_mag_means.append(np.nan)
#             f2_mag_means.append(np.nan)
#             f1_f0_ratios_mag_means.append(np.nan)
#             f2_f0_ratios_mag_means.append(np.nan)
#             continue
        
#         # Calcola statistiche
#         f0_means.append(np.mean(f0_call))
#         f0_stds.append(np.std(f0_call))
#         f0_skewnesses.append(stats.skew(f0_call))
#         f0_kurtosises.append(stats.kurtosis(f0_call))
#         f0_bandwidths.append(np.max(f0_call) - np.min(f0_call))
#         f0_fst_order_diffs_means.append(np.diff(f0_call).mean())
#         f0_slopes.append((f0_call[-1] - f0_call[0]) / (len(f0_call) * hop_length / sr))
#         f0_mag_means.append(np.mean(f0_call))  # Puoi adattare questo se hai un calcolo specifico

#         # Calcola F1 e F2
#         f1_mag_means.append(np.mean(f0_call) * 2)  # Frequenza 2 volte F0
#         f2_mag_means.append(np.mean(f0_call) * 3)  # Frequenza 3 volte F0
#         f1_f0_ratios_mag_means.append(f1_mag_means[-1] / (f0_mag_means[-1] if f0_mag_means[-1] > 0 else np.nan))
#         f2_f0_ratios_mag_means.append(f2_mag_means[-1] / (f0_mag_means[-1] if f0_mag_means[-1] > 0 else np.nan))

#     # Crea un DataFrame per i risultati
#     f0_features_calls = pd.DataFrame({
#         'F0 Mean': f0_means,
#         'F0 Std': f0_stds,
#         'F0 Skewness': f0_skewnesses,
#         'F0 Kurtosis': f0_kurtosises,
#         'F0 Bandwidth': f0_bandwidths,
#         'F0 1st Order Diff': f0_fst_order_diffs_means,
#         'F0 Slope': f0_slopes,
#         'F0 Mag Mean': f0_mag_means,
#         'F1 Mag Mean': f1_mag_means,
#         'F2 Mag Mean': f2_mag_means,
#         'F1-F0 Ratio': f1_f0_ratios_mag_means,
#         'F2-F0 Ratio': f2_f0_ratios_mag_means,
#     })

#     return f0_features_calls
#  offliine n_fft=4096, pyin_fmin_hz=2000, pyin_fmax_hz=12500,  frame_length=2048                                                              
def compute_f0_features_realtime(audio_data, onsets, offsets, sr, hop_length, frame_length=1024, n_fft=2048, 
                             pyin_fmin_hz=2000, pyin_fmax_hz=12500, pyin_beta=(0.10, 0.10), 
                             pyin_ths=100, pyin_resolution=0.02):
    """
    Calcola le caratteristiche della frequenza fondamentale in tempo reale per segmenti audio definiti da onsets e offsets.
    Implementazione allineata con la versione offline per garantire risultati coerenti.
    
    Args:
        audio_data: Audio da analizzare
        onsets: Lista di tempi di inizio delle chiamate (in secondi)
        offsets: Lista di tempi di fine delle chiamate (in secondi)
        sr: Frequenza di campionamento
        hop_length: Hop length per la STFT
        frame_length: Lunghezza della finestra per la STFT
        n_fft: Dimensione FFT
        pyin_fmin_hz: Frequenza minima per PYIN
        pyin_fmax_hz: Frequenza massima per PYIN
        pyin_beta: Parametro beta per PYIN (tupla di due valori)
        pyin_ths: Soglia per PYIN
        pyin_resolution: Risoluzione per PYIN
        
    Returns:
        f0_features_calls: DataFrame con le caratteristiche F0 per ogni chiamata
    """

    
    # Assicurati che pyin_beta sia una tupla di due elementi
    if isinstance(pyin_beta, (float, int)):
        pyin_beta = (pyin_beta, pyin_beta)

    print(f"DEBUG: Rilevati {len(onsets)} onsets, {len(offsets)} offsets")
    
    # Inizializza le liste per i risultati
    call_numbers = []
    F0_means = []
    F0_stds = []
    F0_skewnesses = []
    F0_kurtosises = []
    F0_bandwidths = []
    F0_fst_order_diffs_means = []
    F0_slopes = []
    F0_mag_means = []
    F1_mag_means = []
    F2_mag_means = []
    F1_F0_ratios_mag_means = []
    F2_F0_ratios_mag_means = []
    
    # Stima il pitch usando PYIN
    f0_pyin_lb, _, _ = lb.pyin(audio_data, sr=sr, frame_length=frame_length, 
                            hop_length=hop_length, fmin=pyin_fmin_hz, fmax=pyin_fmax_hz, 
                            n_thresholds=int(pyin_ths), beta_parameters=pyin_beta, 
                            resolution=pyin_resolution)
    
    # Calcola lo spettrogramma
    S = np.abs(lb.stft(y=audio_data, n_fft=frame_length, hop_length=hop_length))
    
    # Segmenta lo spettrogramma in chiamate
    calls_S = []
    for start, end in zip(onsets, offsets):
        # Converti tempi in frames
        start_frame = int(start * sr / hop_length)
        end_frame = int(end * sr / hop_length)
        
        # Estrai lo spettrogramma della chiamata
        if end_frame <= S.shape[1]:  # Controlla che l'indice finale non superi la dimensione dello spettrogramma
            call_s = S[:, start_frame:end_frame]
            calls_S.append(call_s)
        else:
            calls_S.append(np.zeros((S.shape[0], 1)))  # Aggiungi uno spettrogramma vuoto se fuori dai limiti
    
    # Segmenta F0 in chiamate
    f0_calls = []
    for start, end in zip(onsets, offsets):
        start_frame = int(start * sr / hop_length)
        end_frame = int(end * sr / hop_length)
        
        if end_frame <= len(f0_pyin_lb):  # Controlla che l'indice finale non superi la lunghezza di f0
            f0_call = f0_pyin_lb[start_frame:end_frame]
            f0_calls.append(f0_call)
        else:
            f0_calls.append(np.array([np.nan]))  # Aggiungi una chiamata vuota se fuori dai limiti
    
    discarded_calls = 0
    # Elabora ogni chiamata
    for i, (f0_call, call_s) in enumerate(zip(f0_calls, calls_S)):
        call_numbers.append(i)
        
        # Rimuovi NaN dalla chiamata F0
        f0_call_without_nans = f0_call[~np.isnan(f0_call)]
        
        # Verifica se la chiamata è vuota
        if len(f0_call_without_nans) == 0:
            discarded_calls += 1
            F0_means.append(np.nan)
            F0_stds.append(np.nan)
            F0_skewnesses.append(np.nan)
            F0_kurtosises.append(np.nan)
            F0_bandwidths.append(np.nan)
            F0_fst_order_diffs_means.append(np.nan)
            F0_slopes.append(np.nan)
            F0_mag_means.append(np.nan)
            F1_mag_means.append(np.nan)
            F2_mag_means.append(np.nan)
            F1_F0_ratios_mag_means.append(np.nan)
            F2_F0_ratios_mag_means.append(np.nan)
        else:
            # Calcola le statistiche
            f0_call_mean = f0_call_without_nans.mean()
            f0_call_std = f0_call_without_nans.std()
            f0_call_skewness = stats.skew(f0_call_without_nans)
            f0_call_kurtosis = stats.kurtosis(f0_call_without_nans)
            
            # Calcola bandwidth
            min_f0 = np.min(f0_call_without_nans)
            max_f0 = np.max(f0_call_without_nans)
            f0_bandwidth = max_f0 - min_f0
            
            # Calcola la derivata prima dell'F0
            f0_fst_order_diff_mean = np.diff(f0_call_without_nans).mean()
            
            # Calcola la pendenza dell'F0 come nella versione offline
            onset_in_time = 0
            peak_index = np.argmax(f0_call_without_nans)
            attack_fr_hz = f0_call_without_nans[peak_index] - f0_call_without_nans[0]
            peak_index_in_time = lb.frames_to_time(peak_index, sr=sr, hop_length=hop_length)
            attack_time_f0_hz = peak_index_in_time - onset_in_time
            
            if attack_time_f0_hz == 0:
                F0_slope = 0
            elif np.isnan(attack_fr_hz) or attack_time_f0_hz < 0:
                F0_slope = np.nan
            else:
                F0_slope = attack_fr_hz / attack_time_f0_hz
            
            # Calcola F1 e F2
            F1_Hz_withoutNans = f0_call_without_nans * 2
            F2_Hz_withoutNans = f0_call_without_nans * 3
            
            # Calcola le magnitudini di F0, F1 e F2
            F0_mag = []
            F1_mag = []
            F2_mag = []
            
            for time_frame, freqhz in enumerate(f0_call):
                if np.isnan(freqhz):
                    continue
                
                f0_bin = int(np.floor(freqhz * n_fft / sr))
                f1_bin = int(np.floor(freqhz * 2 * n_fft / sr))
                f2_bin = int(np.floor(freqhz * 3 * n_fft / sr))
                
                # Controlla che gli indici siano validi per lo spettrogramma della chiamata
                if f0_bin < call_s.shape[0] and time_frame < call_s.shape[1]:
                    F0_mag.append(call_s[f0_bin, time_frame])
                
                try:
                    if time_frame < call_s.shape[1]:
                        F1_mag.append(call_s[f1_bin, time_frame])
                except IndexError:
                    F1_mag.append(0)
                
                try:
                    if time_frame < call_s.shape[1]:
                        F2_mag.append(call_s[f2_bin, time_frame])
                except IndexError:
                    F2_mag.append(0)
            
            # Rimuovi NaN dalle magnitudini
            F1_mag_without_nans = [mag for mag in F1_mag if not np.isnan(mag)]
            F2_mag_without_nans = [mag for mag in F2_mag if not np.isnan(mag)]
            
            # Calcola le medie delle magnitudini
            F0_mag_mean = np.mean(F0_mag) if F0_mag else np.nan
            
            if not F1_mag_without_nans:
                F1_mag_mean = np.nan
            else:
                F1_mag_mean = np.mean(F1_mag_without_nans)
            
            if not F2_mag_without_nans:
                F2_mag_mean = np.nan
            else:
                F2_mag_mean = np.mean(F2_mag_without_nans)
            
            # Calcola i rapporti delle magnitudini
            if np.isnan(F1_mag_mean):
                F1_F0_ratios_magnitude_mean = np.nan
            else:
                F1_F0_ratios_magnitude = [F1 / F0 for F1, F0 in zip(F1_mag, F0_mag) if F0 > 0]
                F1_F0_ratios_magnitude_mean = np.mean(F1_F0_ratios_magnitude) if F1_F0_ratios_magnitude else np.nan
            
            if np.isnan(F2_mag_mean):
                F2_F0_ratios_magnitude_mean = np.nan
            else:
                F2_F0_ratios_magnitude = [F2 / F0 for F2, F0 in zip(F2_mag, F0_mag) if F0 > 0]
                F2_F0_ratios_magnitude_mean = np.mean(F2_F0_ratios_magnitude) if F2_F0_ratios_magnitude else np.nan
            
            # Aggiungi alle liste dei risultati
            F0_means.append(f0_call_mean)
            F0_stds.append(f0_call_std)
            F0_skewnesses.append(f0_call_skewness)
            F0_kurtosises.append(f0_call_kurtosis)
            F0_bandwidths.append(f0_bandwidth)
            F0_fst_order_diffs_means.append(f0_fst_order_diff_mean)
            F0_slopes.append(F0_slope)
            F0_mag_means.append(F0_mag_mean)
            F1_mag_means.append(F1_mag_mean)
            F2_mag_means.append(F2_mag_mean)
            F1_F0_ratios_mag_means.append(F1_F0_ratios_magnitude_mean)
            F2_F0_ratios_mag_means.append(F2_F0_ratios_magnitude_mean)
    
    # Crea un DataFrame per i risultati
    f0_features_calls = pd.DataFrame({
        'Call Number': call_numbers,
        'F0 Mean': F0_means,
        'F0 Std': F0_stds,
        'F0 Skewness': F0_skewnesses,
        'F0 Kurtosis': F0_kurtosises,
        'F0 Bandwidth': F0_bandwidths,
        'F0 1st Order Diff': F0_fst_order_diffs_means,
        'F0 Slope': F0_slopes,
        'F0 Mag Mean': F0_mag_means,
        'F1 Mag Mean': F1_mag_means,
        'F2 Mag Mean': F2_mag_means,
        'F1-F0 Ratio': F1_F0_ratios_mag_means,
        'F2-F0 Ratio': F2_F0_ratios_mag_means
    })
    
    print(f"Discarded calls: {discarded_calls}")
    
    return f0_features_calls


def get_calls_waveform(audio_data, onsets, offsets, sr):
    """Extract waveform segments corresponding to call onsets/offsets."""
    calls = []
    for onset, offset in zip(onsets, offsets):
        start_sample = int(onset * sr)
        end_sample = int(offset * sr)
        calls.append(audio_data[start_sample:end_sample])
    return calls


import numpy as np
from scipy import signal

def bp_filter(audio_y, sr=44100, lowcut=2050, highcut=13000):
    """
    Apply a bandpass filter to the audio signal.

    Parameters:
    - audio_y: The input audio signal (numpy array).
    - sr: The sampling rate of the audio signal.
    - lowcut: The lower cutoff frequency for the bandpass filter.
    - highcut: The upper cutoff frequency for the bandpass filter.

    Returns:
    - The filtered audio signal.
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist        
    high = highcut / nyquist
    
    # Design the bandpass filter using a Butterworth filter
    b, a = signal.butter(5, [low, high], btype='band')
    
    # Apply the filter using filtfilt for zero-phase filtering
    filtered_audio = signal.filtfilt(b, a, audio_y)
    
    return filtered_audio


def segment_spectrogram(spectrogram, onsets, offsets, sr=44100):

    # Initialize lists to store spectrogram slices
    calls_S = []
    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Convert time (in seconds) to sample indices
        onset_frames = lb.time_to_frames(onset, sr=sr)
        offset_frames = lb.time_to_frames(offset, sr=sr)

        #Extract the spectrogram slice from onset to offset 
        # REVIEW THIS value of epsilon
        # epsilon = duration*0.0001
        # epsilon_samples = lb.time_to_samples(epsilon, sr=44100)

        call_spec = spectrogram[:, onset_frames: offset_frames]#+ epsilon_samples]

        # Append the scaled log-spectrogram slice to the calls list
        calls_S.append(call_spec)
    
    return calls_S