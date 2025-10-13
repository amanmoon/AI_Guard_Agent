import eel
import time
from .ASR.captureAudio import RealTimeAudioProcessor
from .CV.recognizeFaces import FacialRecognizer
from .NLP.gemma import Gemma

TRUSTED_FACES_DIR = "trusted_faces"
MODEL_TO_USE = "hog"

@eel.expose
def start_listening():
    """Called from JavaScript to start the whole process."""
    
    gemma_chat = Gemma()
    audio_processor = RealTimeAudioProcessor(model_size="base.en", llm_chat=gemma_chat.chat)
    image_recognizer = FacialRecognizer(trusted_faces_dir=TRUSTED_FACES_DIR, model=MODEL_TO_USE)  

    # --- Start the System ---
    audio_processor.start()

    try:
        # Main loop to manage facial recognition based on guard mode
        while audio_processor.is_running:
            if audio_processor.guard_mode and image_recognizer._running:
                gemma_chat.set_verification_status(image_recognizer.is_verified)
            if audio_processor.guard_mode and not image_recognizer._running:
                image_recognizer.start_recognition()
            
            elif not audio_processor.guard_mode and image_recognizer._running:
                image_recognizer.stop_recognition()
                gemma_chat.set_verification_status(True)
            time.sleep(1) # Check the status every second.

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down all systems.")
    
    finally:
        # Gracefully stop all threads and processes
        if image_recognizer._running:
            image_recognizer.stop_recognition()
        audio_processor.stop()
        print("System shutdown complete.")
    