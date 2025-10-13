import face_recognition
import cv2
import os
import time
import threading

class FacialRecognizer:
    """
    A class to handle facial recognition tasks in a separate thread.

    This class loads known faces, processes video frames to detect and identify faces,
    and determines if a trusted individual is present. The recognition loop runs in
    a background thread.
    """

    def __init__(self, trusted_faces_dir, tolerance=0.6, model="hog"):
        """
        Initializes the FacialRecognizer.

        Args:
            trusted_faces_dir (str): The path to the directory containing images of trusted faces.
            tolerance (float): How much distance between faces to consider it a match.
                               Lower is stricter. 0.6 is a good default.
            model (str): The face detection model to use. Options are "hog" (faster, less accurate)
                         or "cnn" (slower, more accurate, better for angled faces).
        """
        self.trusted_faces_dir = trusted_faces_dir
        self.trusted_face_encodings = []
        self.trusted_face_names = []
        self.tolerance = tolerance
        self.model = model
        
        # --- State Variables ---
        self.is_verified = False 
        self.untrusted_start_time = None 
        
        # --- Threading Control ---
        self._running = False
        self._recognition_thread = None
        
        print(f"Loading trusted faces... (using '{self.model}' model for detection)")
        self._load_trusted_faces()

    def _load_trusted_faces(self):
        """
        Loads and encodes faces from the trusted faces directory.
        """
        if not os.path.isdir(self.trusted_faces_dir):
            return 

        for filename in os.listdir(self.trusted_faces_dir):
            filepath = os.path.join(self.trusted_faces_dir, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image = face_recognition.load_image_file(filepath)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        self.trusted_face_encodings.append(face_encodings[0])
                        self.trusted_face_names.append(os.path.splitext(filename)[0])
                        print(f"Successfully loaded and encoded: {filename}")
                    else:
                        print(f"Warning: No faces found in {filename}. Skipping.")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        if not self.trusted_face_encodings:
            print("\nWarning: No trusted faces were loaded. The system will not recognize anyone.")
        else:
            print(f"\nFinished loading {len(self.trusted_face_encodings)} trusted faces.")

    def process_frame(self, frame):
        """
        Processes a single video frame to detect and recognize faces.

        Args:
            frame (np.ndarray): The video frame (from OpenCV).

        Returns:
            str: The overall status ('Trusted', 'Untrusted', 'No Face Detected').
        """
        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model=self.model)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        overall_status = "No Face Detected"
        if face_encodings:
            is_trusted_person_present = any(True in face_recognition.compare_faces(self.trusted_face_encodings, enc, self.tolerance) for enc in face_encodings)
            overall_status = "Trusted" if is_trusted_person_present else "Untrusted"
        return overall_status

    def _recognition_loop(self):
        """The main recognition loop that runs once every 10 seconds."""
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("\nError: Could not open video stream.")
            return

        last_verified_state = self.is_verified
        print(f"Initial State: {'Verified' if self.is_verified else 'Unverified'}")
        
        while self._running:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame.")
                time.sleep(1) 
                continue
            
            status = self.process_frame(frame)
            if status == "Trusted":
                self.is_verified = True
                self.untrusted_start_time = None
            else: 
                if self.untrusted_start_time is None:
                    self.untrusted_start_time = time.time()
                elif time.time() - self.untrusted_start_time > 10:
                    self.is_verified = False

            if self.is_verified != last_verified_state:
                print(f"State changed to: {'Verified' if self.is_verified else 'Unverified'}")
                last_verified_state = self.is_verified
            
            time.sleep(1)
        
        video_capture.release()
        print("Recognition loop stopped and camera released.")


    def start_recognition(self):
        """
        Starts the recognition process in a background thread.
        """
        if not os.path.isdir(self.trusted_faces_dir) or not any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(self.trusted_faces_dir)):
            print(f"\n--- IMPORTANT ---")
            print(f"Directory '{self.trusted_faces_dir}' does not exist or is empty.")
            print("Please create it and add images of trusted people.")
            print("-----------------")
            return
        
        if self._running:
            print("Recognition is already running.")
            return

        self._running = True
        self._recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._recognition_thread.start()
        print("\nRecognition thread started.")

    def stop_recognition(self):
        """Stops the recognition process."""
        if not self._running:
            print("Recognition is not running.")
            return

        print("\nStopping recognition process...")
        self._running = False
        self._recognition_thread.join() 
        print("Recognition thread stopped.")
