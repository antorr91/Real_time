import sounddevice as sd 
import numpy as np
import madmom
import joblib
import librosa as lb
import csv
import os
import pandas as pd
import warnings
from Onset_offset_real_time import (detect_onset_offset, is_duplicate_event)  
from features_realtime import (spectral_centroid_realtime, 
    rms_features_realtime,
    compute_envelope_features_realtime,
    compute_f0_features_realtime, get_calls_waveform, bp_filter
)
import shutil
from pathlib import Path
import queue
import threading
import json
from datetime import datetime


warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
import datetime
# Audio parameters
duration = 10  # Duration of recording in seconds
sample_rate = 44100  # Sampling frequency
buffer_size = 1024  #2048 Buffer size
hop_length = 512 # Hop length from offline study

# Parameters specific to chick vocalizations (from offline data)
MIN_DURATION = 0.01  # Minimum duration of a vocalization (0.02 in offline data)
MAX_DURATION = 0.54  # Maximum duration (from offline data)
AV_DURATION = 0.19   # Average duration (from offline data)

# Spectral parameters for chicks (from offline data)
SPEC_NUM_BANDS = 15
# SPEC_FMIN = 2500
# SPEC_FMAX = 5000


SPEC_FMIN = 100
SPEC_FMAX = 12600
SPEC_FREF = 2800
PP_THRESHOLD = 1.8  # Onset threshold (from offline data)
PP_PRE_AVG = 25
PP_POST_AVG = 1
PP_PRE_MAX = 3
PP_POST_MAX = 2

# Parameters for offset detection
OFFSET_FMIN = 2050
OFFSET_FMAX = 8000
OFFSET_N_MELS = 15



ENABLE_FILTER = False  # Set to True to enable bandpass filtering
# Rimuovi questa definizione iniziale
# csv_filename = os.path.join("C:\\Users\\anton\\Real_time", "chick_calls_detection.csv")

save_directory = "C:\\Users\\anton\\Real_time\\features"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Definisci la funzione per il nome del file con timestamp
def get_timestamp_filename():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f'recording_{timestamp}.csv'

# Inizializza il nome del file con timestamp nella directory corretta
csv_filename = os.path.join(save_directory, get_timestamp_filename())

# Inizializza il contatore di chiamate
call_counter = 1

# Scrivi le intestazioni CSV nel file corretto
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Onset (s)", "Offset (s)", "Duration (s)", "Call Number", 
                     "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis",
                     "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope",
                     "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", 
                     "F2-F0 Ratio", "Spectral Centroid Mean",
                     "Spectral Centroid Std", "RMS Mean", "RMS Std",
                     "Slope", "Attack_magnitude", "Attack_time"])

# print("Dispositivi audio disponibili:")
# print(sd.query_devices())
# Imposta il microfono corretto
sd.default.device = 1  # Cambia a 1 per usare il "Microphone Array"

# Global variables to maintain state between blocks
global_time = 0
audio_buffer = np.array([])
last_processed_time = 0
detection_buffer = []  # Buffer to avoid duplicates

# Configurazione del salvataggio
class SaveConfig:
    def __init__(self, base_dir="C:\\Users\\anton\\Real_time"):
        self.base_dir = Path(base_dir)
        self.features_dir = self.base_dir / "features"
        self.backup_dir = self.base_dir / "backups"
        self.config_file = self.base_dir / "save_config.json"
        self.buffer_size = 10  # Numero di eventi prima del flush
        self.setup_directories()
        
    def setup_directories(self):
        """Crea le directory necessarie se non esistono."""
        for directory in [self.features_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def get_current_filename(self):
        """Genera il nome del file con timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return self.features_dir / f'recording_{timestamp}.csv'
    
    def create_backup(self, source_file):
        """Crea un backup del file."""
        if source_file.exists():
            backup_file = self.backup_dir / f"{source_file.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            shutil.copy2(source_file, backup_file)
            return backup_file
        return None

class DataSaver:
    def __init__(self, config):
        self.config = config
        self.current_file = config.get_current_filename()
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        self._initialize_csv()
        
    def _initialize_csv(self):
        """Inizializza il file CSV con le intestazioni."""
        try:
            with open(self.current_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Onset (s)", "Offset (s)", "Duration (s)", "Call Number", 
                               "F0 Mean", "F0 Std", "F0 Skewness", "F0 Kurtosis",
                               "F0 Bandwidth", "F0 1st Order Diff", "F0 Slope",
                               "F1 Mag Mean", "F2 Mag Mean", "F1-F0 Ratio", 
                               "F2-F0 Ratio", "Spectral Centroid Mean",
                               "Spectral Centroid Std", "RMS Mean", "RMS Std",
                               "Slope", "Attack_magnitude", "Attack_time",
                               "Timestamp"])
        except Exception as e:
            print(f"Errore nell'inizializzazione del CSV: {e}")
            raise

    def add_data(self, features_df):
        """Aggiunge dati al buffer e li salva quando necessario."""
        with self.buffer_lock:
            features_df['Timestamp'] = datetime.now().isoformat()
            self.buffer.append(features_df)
            if len(self.buffer) >= self.config.buffer_size:
                self._flush_buffer()

    def _flush_buffer(self):
        """Invia i dati dal buffer alla coda di salvataggio."""
        if self.buffer:
            combined_df = pd.concat(self.buffer, ignore_index=True)
            self.save_queue.put(combined_df)
            self.buffer.clear()

    def _save_worker(self):
        """Worker thread per il salvataggio dei dati."""
        while True:
            try:
                df = self.save_queue.get()
                if df is None:
                    break
                
                try:
                    df.to_csv(self.current_file, mode='a', header=False, index=False)
                    # Crea backup periodico
                    if self.current_file.stat().st_size > 1024 * 1024:  # 1MB
                        self.config.create_backup(self.current_file)
                except Exception as e:
                    print(f"Errore nel salvataggio dei dati: {e}")
                    # Tenta di creare un nuovo file in caso di errori
                    self.current_file = self.config.get_current_filename()
                    self._initialize_csv()
                    df.to_csv(self.current_file, mode='a', header=False, index=False)
                
                self.save_queue.task_done()
            except Exception as e:
                print(f"Errore nel worker di salvataggio: {e}")

    def close(self):
        """Chiude il saver e salva eventuali dati rimanenti."""
        self._flush_buffer()
        self.save_queue.put(None)
        self.save_thread.join()

# Inizializzazione del sistema di salvataggio
save_config = SaveConfig()
data_saver = DataSaver(save_config)

# Versione aggiornata della funzione process_audio
def process_audio(indata, frames, time, status):
    global audio_buffer, global_time, last_processed_time, detection_buffer, call_counter
    print(f"Process_audio called - buffer len: {len(audio_buffer)}")

    if status:
        print(f"Status: {status}")
    
    # Check if the audio input is empty
    current_audio = np.squeeze(indata)

    if current_audio.size == 0:
        print("Empty audio input detected.")
        return
    
    if ENABLE_FILTER:
        # Apply bandpass filter if enabled
        filtered_audio = bp_filter(current_audio, sr=sample_rate)
    else:
        filtered_audio = current_audio

   
    audio_buffer = np.concatenate((audio_buffer, filtered_audio))

    # print(f"Audio buffer length: {len(audio_buffer)}")
    
    # Update global time
    block_duration = frames / sample_rate
    global_time += block_duration
    
    # Process only if we have enough data
    min_buffer_size = 1.5 * sample_rate  # Minimum size to have enough data for analysis
    if len(audio_buffer) >= min_buffer_size:
        # Calculate start timestamp for this block
        start_time = global_time - (len(audio_buffer) / sample_rate)
        
        # Detect onset and offset
        onsets_seconds, offsets_seconds = detect_onset_offset(audio_buffer, sample_rate, start_time)

        print(f"Detected {len(onsets_seconds)} onsets and {len(offsets_seconds)} offsets.") 
        
        # Filter out events that have already been detected
        new_events = []
        for onset, offset in zip(onsets_seconds, offsets_seconds):
            if onset > last_processed_time and not is_duplicate_event(onset, detection_buffer):
                duration = offset - onset
                # Check if duration is within expected limits
                if MIN_DURATION <= duration <= MAX_DURATION:
                    new_events.append((onset, offset, duration))
                    # Add to deduplication buffer
                    detection_buffer.append((onset, offset))
        
        # Limit size of deduplication buffer
        while len(detection_buffer) > 20:
            detection_buffer.pop(0)
            
        # Update last processed time
        if len(onsets_seconds) > 0:
            last_processed_time = max(last_processed_time, max(onsets_seconds))
        
        if new_events:
            print(f"Detected {len(new_events)} new calls:")
            for onset, offset, duration in new_events:
                print(f"  Onset: {onset:.3f}s, Offset: {offset:.3f}s, Duration: {duration:.3f}s")

            # Estrai onsets e offsets relativi
            onsets = np.array([ev[0] - start_time for ev in new_events])
            offsets = np.array([ev[1] - start_time for ev in new_events])

            # Estrai waveform delle calls
            calls_y = get_calls_waveform(audio_buffer, onsets, offsets, sr=sample_rate)

            # Calcola le features
            envelope_features = compute_envelope_features_realtime(calls_y, sr=sample_rate)

            rms_mean, rms_std = rms_features_realtime(calls_y, 2048, hop_length)


            mel_spec = lb.feature.melspectrogram(y=audio_buffer, sr=sample_rate, n_fft= 2048,
                                hop_length=512, n_mels=128, fmin=2000, fmax=12600)
            
            sc_mean, sc_std = spectral_centroid_realtime(mel_spec, onsets, offsets, sample_rate, frame_length=2048, hop_length=hop_length)
  

            # print(f"Envelope features: {envelope_features}")
            f0_features =compute_f0_features_realtime(audio_buffer, onsets, offsets, sample_rate, hop_length, frame_length=1024, n_fft=2048, 
                             pyin_fmin_hz=2000, pyin_fmax_hz=12500, pyin_beta=(0.10, 0.10), 
                             pyin_ths=100, pyin_resolution=0.02)
    
    
          
            
            # Crea la lista di numeri di chiamata per questi eventi
            call_numbers = list(range(call_counter, call_counter + len(new_events)))
            call_counter += len(new_events)

            # Assicurati che tutte le feature F0 siano disponibili o aggiungi valori predefiniti
            f0_features_dict = f0_features.to_dict(orient='list')
            f0_required_features = [
                'F0 Mean', 'F0 Std', 'F0 Skewness', 'F0 Kurtosis',
                'F0 Bandwidth', 'F0 1st Order Diff', 'Slope',
                'F1 Mag Mean', 'F2 Mag Mean', 'F1-F0 Ratio',
                'F2-F0 Ratio'
            ]
            for feature in f0_required_features:
                if feature not in f0_features_dict:
                    # Se manca una feature richiesta, aggiungi un valore predefinito (0 o NaN)
                    f0_features_dict[feature] = [np.nan] * len(new_events)

            # Crea un DataFrame per le features con l'ordine esatto richiesto
            features_df = pd.DataFrame({
                'onsets_sec': [ev[0] for ev in new_events],
                'offsets_sec': [ev[1] for ev in new_events],
                'Duration_call': [ev[2] for ev in new_events],
                'Call Number': call_numbers,
                'F0 Mean': f0_features_dict.get('F0 Mean', [np.nan] * len(new_events)),
                'F0 Std': f0_features_dict.get('F0 Std', [np.nan] * len(new_events)),
                'F0 Skewness': f0_features_dict.get('F0 Skewness', [np.nan] * len(new_events)),
                'F0 Kurtosis': f0_features_dict.get('F0 Kurtosis', [np.nan] * len(new_events)),
                'F0 Bandwidth': f0_features_dict.get('F0 Bandwidth', [np.nan] * len(new_events)),
                'F0 1st Order Diff': f0_features_dict.get('F0 1st Order Diff', [np.nan] * len(new_events)),
                'F0 Slope': f0_features_dict.get('Slope', [np.nan] * len(new_events)),
                'F0 Mag Mean': f0_features_dict.get('F0 Mag Mean', [np.nan] * len(new_events)),
                'F1 Mag Mean': f0_features_dict.get('F1 Mag Mean', [np.nan] * len(new_events)),
                'F2 Mag Mean': f0_features_dict.get('F2 Mag Mean', [np.nan] * len(new_events)),
                'F1-F0 Ratio': f0_features_dict.get('F1-F0 Ratio', [np.nan] * len(new_events)),
                'F2-F0 Ratio': f0_features_dict.get('F2-F0 Ratio', [np.nan] * len(new_events)),
                'Spectral Centroid Mean': sc_mean,
                'Spectral Centroid Std': sc_std,
                'RMS Mean': rms_mean,
                'RMS Std': rms_std,
                'Slope': envelope_features.to_dict(orient='list').get('Slope', [np.nan] * len(new_events)),
                'Attack_magnitude': envelope_features.to_dict(orient='list').get('Attack_magnitude', [np.nan] * len(new_events)),
                'Attack_time': envelope_features.to_dict(orient='list').get('Attack_time', [np.nan] * len(new_events))
            })

            # Stampa le features estratte
            for idx, row in features_df.iterrows():
                print(f"  Onset: {row['onsets_sec']:.3f}s, Offset: {row['offsets_sec']:.3f}s, Duration: {row['Duration_call']:.3f}s, "
                    f"Call Number: {row['Call Number']}, F0 Mean: {row['F0 Mean']:.2f}, "
                    f"Spectral Centroid Mean: {row['Spectral Centroid Mean']:.2f}, RMS Mean: {row['RMS Mean']:.4f}")

            # Modifica il salvataggio per utilizzare il nuovo sistema
            data_saver.add_data(features_df)

            # Keep only the last second of audio for overlap
            overlap_samples = int(1 * sample_rate)
            if len(audio_buffer) > overlap_samples:
                audio_buffer = audio_buffer[-overlap_samples:]

# Aggiorna il blocco try-except per usare il nuovo nome file
try:
    print(f"Start recording for {duration} seconds...")
    print(f"The results will be saved in: {save_config.features_dir}")
    print("Parameters used:")
    print(f"  Minimum call duration: {MIN_DURATION} s")
    print(f"  Maximum call duration: {MAX_DURATION} s")
    print(f"Average call duration: {AV_DURATION} s")
    print(f"  Onset threshold: {PP_THRESHOLD}")
    print(f"  Frequency band: {SPEC_FMIN}-{SPEC_FMAX} Hz")
    
    # Initialise the audio stream
    with sd.InputStream(callback=process_audio, channels=1, samplerate=sample_rate, blocksize=buffer_size):
        # Keep the programme running for the specified duration
        sd.sleep(int(duration * 1000))
    
    print("Recording completed. Results saved in", save_config.features_dir)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Chiudi correttamente il sistema di salvataggio
    data_saver.close()