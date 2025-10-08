import pyaudio
import threading
import time
import numpy as np
import whisper
import torch
import wave


class Listner:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, output=True, frames_per_buffer=1024)
        print("Audio Listner Initialized")
    
    def listen(self, queue):
        while True:
            data = self.stream.read(16000, exception_on_overflow=False)
            queue.append(data)
    
    def run(self, queue):
        self.listeningThread = threading.Thread(target=self.listen,args=(queue,), daemon=True)
        self.listeningThread.start()

    def terminate(self):
        try:
            time.sleep(0.1)
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            self.listeningThread.join()
            print("Audio Listner Terminated")

class AudioInterpreter:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.audioData = list()
        self.listner = Listner()
        self.model = whisper.load_model("medium.en").to(device)

    def interpret(self):
        print("listening")
        while True:
            audio = self.audioData

            if (len(audio) > 10):
                audio = audio[-10:]

            audio = b''.join(audio)
            
            audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32)

            audio_np = audio_np / 32768.0

            out = self.model.transcribe(audio_np)
            if (out['segments'] and out['segments'][0]['no_speech_prob'] < 0.5):
                print("Interpreted Audio: ", out['text'])
            

    def run(self):
        self.listner.run(self.audioData)
        self.interpetingThread = threading.Thread(target=self.interpret, daemon=True)
        self.interpetingThread.start()

    def terminate(self):
        try:
            time.sleep(0.1)
        finally:
            self.interpetingThread.join()
            self.listner.terminate()
            del self.model
            with torch.no_grad():
                torch.cuda.empty_cache()
            print("Audio Interpretation Terminated")


if __name__ == "__main__":
    try:
        audioInterpreter = AudioInterpreter()
        audioInterpreter.run()

        while True:
            time.sleep(1) 

    except KeyboardInterrupt:
        print("\nStopping interpreter...")
        audioInterpreter.terminate()
