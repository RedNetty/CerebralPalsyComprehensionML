import cv2
import numpy as np
from keras.models import load_model
import time
import json
from datetime import datetime
import threading
import os

class GestureVisualizationSystem:
    def __init__(self):
        # Window names and setup
        self.windows = {
            'main': "Live Camera Input",
            'background': "Background Model",
            'foreground': "Foreground Objects",
            'probability': "Foreground Probability",
            'debug': "Debug Information"
        }
        
        # Settings
        self.settings = {
            'record_video': False,
            'save_frames': False,
            'show_prob_distribution': True,
            'show_fps': True,
            'show_debug': False,
            'detection_threshold': 0.7,
            'gesture_smoothing': True
        }
        
        # Initialize components
        self.setup_model()
        self.setup_windows()
        self.setup_capture()
        self.setup_background_subtractor()
        self.setup_recording()
        
        # Performance tracking
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        
        # Gesture tracking
        self.gesture_history = []
        self.gesture_buffer_size = 5
        self.last_prediction = None
        self.prediction_confidence = 0
        
    def setup_model(self):
        """Load model and gesture mappings"""
        try:
            self.model = load_model('Hand_Gesture_Recognize.h5')
            self.gesture = {
                0: "Left hand up",
                1: "Hands down",
                2: "Right hand up"
            }
            # Load any custom gesture mappings if available
            if os.path.exists('gesture_mapping.json'):
                with open('gesture_mapping.json', 'r') as f:
                    self.gesture.update(json.load(f))
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
    
    def setup_windows(self):
        """Initialize visualization windows"""
        try:
            for name in self.windows.values():
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            
            # Set initial window positions
            screen_width = 1920  # Assuming standard HD resolution
            window_width = screen_width // 2
            cv2.moveWindow(self.windows['main'], 0, 0)
            cv2.moveWindow(self.windows['background'], window_width, 0)
            cv2.moveWindow(self.windows['foreground'], 0, 400)
            cv2.moveWindow(self.windows['probability'], window_width, 400)
        except Exception as e:
            print(f"Error setting up windows: {e}")
            exit(1)
    
    def setup_capture(self):
        """Initialize video capture with optimal settings"""
        self.vc = cv2.VideoCapture(0)
        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.vc.set(cv2.CAP_PROP_FPS, 30)
        self.frame_width = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def setup_background_subtractor(self):
        """Initialize background subtractor with optimized parameters"""
        self.mog = cv2.createBackgroundSubtractorMOG2(
            history=2000,
            varThreshold=16,
            detectShadows=True
        )
        
    def setup_recording(self):
        """Initialize video recording components"""
        self.output_video = None
        self.frame_buffer = []
        self.max_frame_buffer = 300  # 10 seconds at 30 fps
        
    def predict(self, hand):
        """Enhanced prediction with confidence scores"""
        try:
            img = cv2.resize(hand, (50, 50))
            img = np.array(img)
            img = img.reshape((1, 50, 50, 1))
            img = img / 255.0
            
            predictions = self.model.predict(img, verbose=0)
            max_ind = predictions.argmax()
            confidence = float(predictions[0][max_ind])
            
            if confidence < self.settings['detection_threshold']:
                return "No confident prediction", 0
                
            return self.gesture[max_ind], confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error in prediction", 0
            
    def update_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time > 1:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
            
    def process_frame(self, frame):
        """Process frame and apply various image processing steps"""
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Background subtraction pipeline
        fgmask = self.mog.apply(frame)
        fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
        fgdilated = cv2.dilate(
            fgthres,
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=3
        )
        bgmodel = self.mog.getBackgroundImage()
        
        return frame, fgmask, fgthres, fgdilated, bgmodel
        
    def draw_interface(self, frame, prediction, confidence):
        """Draw enhanced UI elements on frame"""
        # Create info panel background
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (240, 240, 240), -1)
        
        # Draw prediction and confidence
        cv2.putText(frame, f"Gesture: {prediction}", (10, 30),
                   cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                   cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
        
        # Draw FPS if enabled
        if self.settings['show_fps']:
            fps_text = f"FPS: {self.fps}"
            cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Draw recording indicator if active
        if self.settings['record_video']:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            
        return frame
        
    def save_frame(self, frame, prediction):
        """Save frame with timestamp and prediction"""
        if self.settings['save_frames']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"frames/gesture_{prediction}_{timestamp}.jpg"
            os.makedirs('frames', exist_ok=True)
            cv2.imwrite(filename, frame)
            
    def run(self):
        """Main application loop"""
        try:
            print("Starting gesture visualization system...")
            print("Press 'q' to quit")
            print("Press 'r' to toggle recording")
            print("Press 's' to save current frame")
            print("Press 'd' to toggle debug view")
            print("Press 'p' to toggle probability distribution")
            
            while True:
                rval, frame = self.vc.read()
                if frame is None:
                    continue
                    
                # Process frame
                frame, fgmask, fgthres, fgdilated, bgmodel = self.process_frame(frame)
                
                # Get prediction
                prediction, confidence = self.predict(fgdilated)
                
                # Smooth predictions if enabled
                if self.settings['gesture_smoothing']:
                    self.gesture_history.append((prediction, confidence))
                    if len(self.gesture_history) > self.gesture_buffer_size:
                        self.gesture_history.pop(0)
                    
                    # Use most common prediction in buffer
                    predictions = [p[0] for p in self.gesture_history]
                    prediction = max(set(predictions), key=predictions.count)
                    confidence = np.mean([p[1] for p in self.gesture_history])
                
                # Update interface
                self.update_fps()
                frame = self.draw_interface(frame, prediction, confidence)
                
                # Display windows
                cv2.imshow(self.windows['main'], frame)
                cv2.imshow(self.windows['background'], bgmodel)
                cv2.imshow(self.windows['foreground'], fgdilated)
                cv2.imshow(self.windows['probability'], fgmask)
                
                # Handle recording
                if self.settings['record_video']:
                    if self.output_video is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"recordings/gesture_recording_{timestamp}.avi"
                        os.makedirs('recordings', exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        self.output_video = cv2.VideoWriter(
                            filename, fourcc, 30.0,
                            (self.frame_width, self.frame_height)
                        )
                    self.output_video.write(frame)
                
                # Handle key presses
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.settings['record_video'] = not self.settings['record_video']
                    if not self.settings['record_video'] and self.output_video:
                        self.output_video.release()
                        self.output_video = None
                elif key == ord('s'):
                    self.save_frame(frame, prediction)
                elif key == ord('d'):
                    self.settings['show_debug'] = not self.settings['show_debug']
                elif key == ord('p'):
                    self.settings['show_prob_distribution'] = not self.settings['show_prob_distribution']
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            # Cleanup
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        if self.output_video:
            self.output_video.release()
        self.vc.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("System shutdown complete")

if __name__ == "__main__":
    app = GestureVisualizationSystem()
    app.run()
