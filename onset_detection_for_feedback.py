import tkinter as tk
import sounddevice as sd
import numpy as np
import madmom
import time
import csv
import threading

# ----- PARAMETERS -----
SAMPLE_RATE = 44100
BUFFER_SIZE = 2048
CALL_THRESHOLD = 10
WINDOW_SEC = 10
EXPERIMENT_DURATION = 6 * 60  # 6 minutes in seconds

# ----- GLOBAL STATE -----
audio_buffer = np.array([])
onset_times = []
start_time = None
running = False

# ----- CSV INIT -----
with open("feedback_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Event_Time_s"])

# ----- DETECTION FUNCTION -----
def detect_onsets(audio, sr, time_offset):
    spec = madmom.audio.spectrogram.FilteredSpectrogram(
        audio,
        sample_rate=sr,
        num_bands=15,
        fmin=2500,
        fmax=5000,
        fref=2800,
        norm_filters=True,
        unique_filters=True
    )
    activation = madmom.features.onsets.high_frequency_content(spec)
    peaks = madmom.features.onsets.peak_picking(activation, threshold=1.8)
    return np.array([p * BUFFER_SIZE / sr + time_offset for p in peaks])

# ----- AUDIO CALLBACK -----
def process_audio(indata, frames, time_info, status):
    global audio_buffer, onset_times, start_time, running

    if not running:
        return

    current_audio = np.squeeze(indata)
    audio_buffer = np.concatenate((audio_buffer, current_audio))

    elapsed_time = time.time() - start_time
    offset = elapsed_time - (len(audio_buffer) / SAMPLE_RATE)

    if len(audio_buffer) >= 1.5 * SAMPLE_RATE:
        onsets = detect_onsets(audio_buffer, SAMPLE_RATE, offset)
        for onset in onsets:
            onset_times.append(onset)

        # Pulisce le onset vecchie
        onset_times[:] = [t for t in onset_times if t >= elapsed_time - WINDOW_SEC]

        # Aggiorna GUI
        call_count = len(onset_times)
        app.update_call_count(call_count)

        if call_count >= CALL_THRESHOLD:
            app.show_feedback_alert()

            # Salva timestamp in CSV
            with open("feedback_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"{elapsed_time:.3f}"])

        audio_buffer = audio_buffer[-SAMPLE_RATE:]

# ----- GUI CLASS -----
class VocalEchoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🐣 Vocal Echo - Manual Feedback Monitor")
        self.root.geometry("400x250")
        self.start_time = None
        self.elapsed_label = tk.Label(root, text="Time: 00:00:000", font=("Arial", 16))
        self.elapsed_label.pack(pady=10)

        self.call_label = tk.Label(root, text="Calls in last 10s: 0", font=("Arial", 16))
        self.call_label.pack(pady=10)

        self.feedback_label = tk.Label(root, text="", font=("Arial", 16), fg="red")
        self.feedback_label.pack(pady=10)

        self.start_btn = tk.Button(root, text="Start", command=self.start_experiment, bg="green", fg="white", font=("Arial", 14))
        self.start_btn.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def start_experiment(self):
        global running, start_time
        self.start_btn.config(state=tk.DISABLED)
        self.feedback_label.config(text="")
        self.call_label.config(text="Calls in last 10s: 0")
        running = True
        start_time = time.time()
        self.start_time = start_time

        self.stream = sd.InputStream(callback=process_audio, channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE)
        self.stream.start()
        self.update_clock()
        threading.Thread(target=self.auto_stop).start()

    def update_clock(self):
        if not running:
            return
        elapsed = time.time() - self.start_time
        if elapsed > EXPERIMENT_DURATION:
            return
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        ms = int((elapsed * 1000) % 1000)
        self.elapsed_label.config(text=f"Time: {mins:02}:{secs:02}:{ms:03}")
        self.root.after(100, self.update_clock)

    def update_call_count(self, count):
        self.call_label.config(text=f"Calls in last 10s: {count}")
        if count < CALL_THRESHOLD:
            self.feedback_label.config(text="")

    def show_feedback_alert(self):
        self.feedback_label.config(text="⚠️ 10 Calls Detected! Give Feedback!")

    def auto_stop(self):
        time.sleep(EXPERIMENT_DURATION)
        self.stop_experiment()

    def stop_experiment(self):
        global running
        running = False
        self.stream.stop()
        self.stream.close()
        self.feedback_label.config(text="Session Ended")

    def quit(self):
        self.stop_experiment()
        self.root.destroy()

# ----- START -----
root = tk.Tk()
app = VocalEchoApp(root)
root.mainloop()
