import eel
import speech_recognition as sr
from fuzzywuzzy import fuzz
import jellyfish
import time
import numpy as np
import threading
import pyaudio
import queue
from collections import deque
import pyttsx3

audio_processor = None

import numpy as np

def analyze_frequency_bands(audio_chunk, bands, rate=16000):
    """
    Calculates the average volume for a specific list of frequency bands.
    
    Args:
        audio_chunk: Raw audio data in bytes.
        bands (list of lists): A list of frequency ranges, e.g., [[200, 1000], [1000, 4000]].
        rate (int): The sample rate of the audio.
        
    Returns:
        list: A list of scaled heights corresponding to the volume of each band.
    """
    try:
        samples = np.frombuffer(audio_chunk, dtype=np.int16)
        if len(samples) == 0:
            return [0] * len(bands)

        # 1. Perform FFT to get frequency magnitudes
        magnitudes = np.abs(np.fft.rfft(samples))
        frequencies = np.fft.rfftfreq(len(samples), d=1. / rate)

        # 2. Calculate the average magnitude for each custom band
        band_magnitudes = []
        for min_freq, max_freq in bands:
            in_band_indices = np.where((frequencies >= min_freq) & (frequencies < max_freq))
            if in_band_indices[0].size > 0:
                avg_magnitude = np.mean(magnitudes[in_band_indices])
                band_magnitudes.append(avg_magnitude)
            else:
                band_magnitudes.append(0)

        # 3. Scale the results for a realistic look
        log_magnitudes = np.log1p(band_magnitudes)
        log_ceiling = 11.5 
        scaled_heights = (log_magnitudes / log_ceiling) * 200
        scaled_heights = np.clip(scaled_heights, 0, 1000)

        return scaled_heights.astype(int).tolist()

    except Exception as e:
        print(f"Error during frequency analysis: {e}")
        return [0] * len(bands)

class RealTimeAudioProcessor:
    """
    Manages a real-time audio stream for simultaneous visualization and speech recognition.
    """
    def __init__(self, model_size="base.en", llm_chat=None):
        self.llm_chat = llm_chat

        try:
            self.tts_engine = pyttsx3.init()
            self.configure_engine()
            self.tts_engine.stop()
            self.tts_engine.say("System initialized and ready.")
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Failed to initialize TTS engine: {e}")
            self.tts_engine = None
            
        # Audio stream parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.SILENCE_THRESHOLD = 0
        self.SILENCE_DURATION_S = 0.5
        self.SILENCE_CHUNKS = int(self.SILENCE_DURATION_S * self.RATE / self.CHUNK)
        self.is_speaking = False
        self.tts_is_speaking = False

        # Speech recognition and TTS
        self.recognizer = sr.Recognizer()
        self.model = model_size

        # State management
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        self.recognition_queue = queue.Queue()
        self.phrase_buffer = deque()
        self.silence_counter = 0
        self.guard_mode = False




        # Target phrases for matching
        self.target_phrases_on = [
            "turn on guard mode", "turn on guard", "switch on guard mode", "switch on guard",
            "enable guard", "start guard mode", "start guard", "activate guard mode",
            "guard on", "activate guard", "turn on the guard", "enable guard mode"
        ]

        self.target_phrases_off = [
            "turn off guard mode", "turn off guard", "switch off guard mode", "switch off guard",
            "disable guard", "stop guard mode", "stop guard", "deactivate guard mode",
            "guard off", "deactivate guard", "turn off the guard", "disable guard mode"
        ]
    
    def calibrate(self, duration_s=2):
        """
        Listens to ambient noise to set a dynamic silence threshold.
        If noise is too high, it will loop and retry until successful.
        """
        print("Starting microphone calibration...")
        while True:
            eel.update_data(f"Calibrating... Please be quiet for {duration_s}s.")
            print(f"\nListening for {duration_s}s of ambient noise...")
            
            temp_stream = self.p.open(format=self.FORMAT,
                                    channels=self.CHANNELS,
                                    rate=self.RATE,
                                    input=True,
                                    frames_per_buffer=self.CHUNK)
            
            noise_levels = []
            num_chunks = int((self.RATE / self.CHUNK) * duration_s)

            for _ in range(num_chunks):
                try:
                    data = temp_stream.read(self.CHUNK, exception_on_overflow=False)
                    volume = np.frombuffer(data, dtype=np.int16).max()
                    noise_levels.append(volume)
                except IOError as e:
                    print(f"Warning: Could not read audio chunk during calibration. {e}")
            
            temp_stream.stop_stream()
            temp_stream.close()

            if not noise_levels:
                print("Calibration failed to capture audio. Retrying in 3 seconds...")
                eel.update_data("Error: Couldn't capture audio. Retrying...")
                time.sleep(3)
                continue 

            base_threshold = max(noise_levels)
            MAX_POSSIBLE_VOLUME = 32767 

            # Check if the noise is too high
            if base_threshold >= MAX_POSSIBLE_VOLUME * 0.9:
                error_msg = "Noise is too high! Please ensure your environment is quiet."
                print(f"WARNING: {error_msg} (Level: {base_threshold}/{MAX_POSSIBLE_VOLUME})")
                print("Recalibrating in 5 seconds...")
                eel.update_data("Noise too high! Please stay quiet. Recalibrating...")
                time.sleep(5) 
                continue

            else:
                # --- SUCCESS CASE ---
                self.SILENCE_THRESHOLD = int(base_threshold * 1.2)
                self.SILENCE_THRESHOLD = max(self.SILENCE_THRESHOLD, 100)
                eel.update_data("Calibration successful! Try speaking now.")
                print(f"\n Calibration successful! Ambient noise level is {base_threshold}.")
                print(f"New SILENCE_THRESHOLD set to: {self.SILENCE_THRESHOLD}")
                break 

    def configure_engine(self):
        """Configures the TTS engine for a more human-like voice."""
        if not self.tts_engine:
            return
        
        # Adjust the speaking rate (slower is often more natural)
        rate = self.tts_engine.getProperty('rate')
        self.tts_engine.setProperty('rate', 150)

        # Change the voice (index 1 is often a female voice on Windows)
        voices = self.tts_engine.getProperty('voices')
        if len(voices) > 1:
            self.tts_engine.setProperty('voice', voices[1].id)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """This function is called by PyAudio for each new audio chunk."""
        bands_to_analyze = [
            [ 400,  800],   
            [ 800, 1600],    
            [1600, 3200],
            [3200, 6400],
            [ 200,  400],    
            [ 100,  200],
            [  50,  100],   
            [6400, 16000]    
        ]
        
        band_volumes = analyze_frequency_bands(
            in_data, 
            bands=bands_to_analyze, 
            rate=self.RATE
        )
        
        shape_multipliers = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        shaped_volumes = [vol * mult for vol, mult in zip(band_volumes, shape_multipliers)]
        v0, v1, v2, v3, v4, v5, v6, v7 = shaped_volumes
        
        final_bars = [
            v7, v6, v5, v4, v3, v2, v1, v0,
            v1, v2, v3, v4, v5, v6, v7 
        ]
        
        eel.update_frequency_bars(final_bars)

        if self.tts_is_speaking:
            return (in_data, pyaudio.paContinue)


        volume = np.frombuffer(in_data, dtype=np.int16).max()
        is_currently_loud = volume > self.SILENCE_THRESHOLD

        if is_currently_loud:
            if not self.is_speaking:
                print("\nSpeech start detected.")
            self.is_speaking = True
            self.silence_counter = 0
            self.phrase_buffer.append(in_data)
        elif self.is_speaking:
            self.phrase_buffer.append(in_data)
            self.silence_counter += 1

            if self.silence_counter > self.SILENCE_CHUNKS:
                complete_phrase_frames = b''.join(list(self.phrase_buffer))
                
                self.phrase_buffer.clear()
                self.silence_counter = 0
                self.is_speaking = False
                
                MIN_PHRASE_CHUNKS = 5 
                if len(complete_phrase_frames) > self.CHUNK * MIN_PHRASE_CHUNKS:
                    print(f"\nPhrase detected! Adding to queue.")
                    self.recognition_queue.put(complete_phrase_frames)
                else:
                    print(f"\nIgnoring short sound burst.")

        return (in_data, pyaudio.paContinue)

    def _get_highest_match_score(self, text_to_check, phrases_to_search):
        """
        Calculates the highest fuzzy/phonetic match score for a text 
        against a list of phrases.
        """
        highest_score = 0
        if not text_to_check:
            return 0

        input_metaphone = ' '.join([jellyfish.metaphone(word) for word in text_to_check.split()])

        for phrase in phrases_to_search:
            text_score = fuzz.token_set_ratio(text_to_check, phrase)
            phrase_metaphone = ' '.join([jellyfish.metaphone(word) for word in phrase.split()])
            phonetic_score = fuzz.ratio(input_metaphone, phrase_metaphone)
            combined_score = (text_score * 0.4) + (phonetic_score * 0.6)

            if combined_score > highest_score:
                highest_score = combined_score
                
        return highest_score
    
    def _recognition_worker(self):
        """Runs in a separate thread, processing phrases from the queue."""
        while self.is_running:
            try:
                # Wait for a complete phrase to be available
                audio_frames = self.recognition_queue.get(timeout=0.1)
                
                audio_data = sr.AudioData(audio_frames, self.RATE, self.p.get_sample_size(self.FORMAT))

                # Recognize using Whisper
                text = self.recognizer.recognize_whisper(
                    audio_data, model=self.model, language="english"
                ).lower().strip()

                print(f"You said: {text}")

                if not text:
                    eel.update_data(text)
                    continue

                MATCH_THRESHOLD = 85

                on_score = self._get_highest_match_score(text, self.target_phrases_on)
                off_score = self._get_highest_match_score(text, self.target_phrases_off)

                print(f"-> On-command score: {on_score:.0f}%, Off-command score: {off_score:.0f}%")

                if on_score > off_score and on_score >= MATCH_THRESHOLD and self.guard_mode == False:
                    print("\nACTION: Turning ON Guard Mode!\n")
                    self.guard_mode = True 
                    eel.trigger_action("Guard Mode Activated!", "green")
                    eel.update_guard_mode(self.guard_mode)
                    try:
                        self.tts_is_speaking = True
                        time.sleep(0.1)
                        self.tts_engine.stop()
                        self.tts_engine.say("Guard mode activated.")
                        self.tts_engine.runAndWait()
                    finally:
                        self.tts_is_speaking = False
                    continue

                elif off_score > on_score and off_score >= MATCH_THRESHOLD and self.guard_mode == True:
                    print("\nACTION: Turning OFF Guard Mode!\n")
                    self.guard_mode = False
                    eel.trigger_action("Guard Mode Deactivated!", "red")
                    eel.update_guard_mode(self.guard_mode)
                    try:
                        self.tts_is_speaking = True
                        time.sleep(0.1)
                        self.tts_engine.stop()
                        self.tts_engine.say("Guard mode deactivated.")
                        self.tts_engine.runAndWait()
                    finally:
                        self.tts_is_speaking = False
                    continue

                else:
                    print("-> No action taken (score below threshold or ambiguous).")

                eel.update_data(text)
                if (len(text) > 10):
                    output = self.llm_chat(text)
                    print(f"Gemma: {output}")
                    try:
                        eel.update_data(f"Guard: {output}")
                        self.tts_is_speaking = True
                        time.sleep(0.1)
                        self.tts_engine.stop()
                        self.tts_engine.say(output)
                        self.tts_engine.runAndWait()
                    finally:
                        self.tts_is_speaking = False

            except queue.Empty:
                continue
            except sr.UnknownValueError:
                print("Whisper could not understand the audio.")
                eel.update_data("Could not understand audio.", "")
            except sr.RequestError as e:
                print(f"Could not request results from Whisper; {e}")
                eel.update_data(f"Whisper API Error: {e}")
            except Exception as e:
                print(f"An error occurred in the recognition worker: {e}")


    def start(self):
        """Starts the audio stream and the recognition worker thread."""
        if self.is_running:
            print("Processor is already running.")
            return

        print("Starting real-time audio processor...")
        self.is_running = True

        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK,
                                  stream_callback=self._audio_callback)
        self.stream.start_stream()

        self.calibrate() 

        self.recognition_thread = threading.Thread(target=self._recognition_worker, daemon=True)
        self.recognition_thread.start()


        print("Microphone stream started. Listening...")

    def stop(self):
        """Stops the audio stream and worker thread gracefully."""
        if not self.is_running:
            return
        
        print("Stopping audio processor...")
        self.is_running = False
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.recognition_thread.join()
        print("Processor stopped.")
