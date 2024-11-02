import subprocess
import cv2
import numpy as np
from keras.models import load_model
import time
import json
import os
from datetime import datetime
import threading
import pyautogui  # for system control functionality

class GestureRecognitionSystem:
    def __init__(self):
        self.windowName = "Advanced Gesture Recognition System"
        self.model = load_model('Hand_Gesture_Recognize.h5')
        
        # Extended gesture dictionary with more gestures
        self.gesture = {
            0: "Left hand up",
            1: "Hands down",
            2: "Right hand up",
            3: "Both hands up",
            4: "Victory sign",
            5: "Thumbs up"
        }
        
        # Settings and states
        self.settings = {
            'muted': True,
            'recording': False,
            'debug_mode': False,
            'gesture_sensitivity': 0.7,
            'system_control': False
        }
        
        # Initialize video capture and processing components
        self.setup_video_capture()
        self.setup_background_subtractor()
        
        # Gesture history for advanced recognition
        self.gesture_history = []
        self.max_history_length = 50
        
        # Performance metrics
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        
        # Recording setup
        self.recording_data = []
        self.output_video = None
        
    def setup_video_capture(self):
        """Initialize video capture with optimal settings"""
        self.vc = cv2.VideoCapture(0)
        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.vc.set(cv2.CAP_PROP_FPS, 30)
        
    def setup_background_subtractor(self):
        """Initialize background subtractor with optimized parameters"""
        self.mog = cv2.createBackgroundSubtractorMOG2(
            history=2000,
            varThreshold=16,
            detectShadows=True
        )
        
    def predict(self, hand):
        """Enhanced prediction with confidence threshold"""
        img = cv2.resize(hand, (50, 50))
        img = np.array(img)
        img = img.reshape((1, 50, 50, 1))
        img = img/255.0
        
        res = self.model.predict(img)
        confidence = np.max(res)
        
        if confidence > self.settings['gesture_sensitivity']:
            max_ind = res.argmax()
            return self.gesture[max_ind], confidence
        return "No confident prediction", 0
    
    def execute_gesture_command(self, gesture):
        """Execute system commands based on recognized gestures"""
        if not self.settings['system_control']:
            return
            
        commands = {
            "Left hand up": lambda: pyautogui.press('volumedown'),
            "Right hand up": lambda: pyautogui.press('volumeup'),
            "Both hands up": lambda: pyautogui.press('space'),
            "Hands down": lambda: pyautogui.press('k'),
        }
        
        if gesture in commands:
            commands[gesture]()
    
    def save_gesture_data(self):
        """Save gesture history to JSON file"""
        if self.recording_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gesture_recording_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(self.recording_data, f)
            
    def update_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        if time.time() - self.fps_start_time > 1:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def draw_interface(self, frame, gesture_text, confidence):
        """Draw enhanced UI elements"""
        # Main info panel
        frame = cv2.rectangle(frame, (0, 0), (800, 100), (153, 204, 255), -1)
        
        # Display gesture and confidence
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30),
                   cv2.FONT_HERSHEY_TRIPLEX, 1, (102, 0, 51))
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                   cv2.FONT_HERSHEY_TRIPLEX, 1, (102, 0, 51))
        
        # Status indicators
        status_y = 130
        statuses = [
            f"FPS: {self.fps}",
            f"Audio: {'Unmuted' if not self.settings['muted'] else 'Muted'}",
            f"Recording: {'On' if self.settings['recording'] else 'Off'}",
            f"System Control: {'Enabled' if self.settings['system_control'] else 'Disabled'}",
            f"Debug Mode: {'On' if self.settings['debug_mode'] else 'Off'}"
        ]
        
        for status in statuses:
            cv2.putText(frame, status, (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            status_y += 20
            
        return frame
    
    def run(self):
        """Main application loop with enhanced functionality"""
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        
        while True:
            rval, frame = self.vc.read()
            if frame is None:
                continue
                
            # Process frame
            frame = cv2.flip(frame, 1)
            fgmask = self.mog.apply(frame)
            fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
            fgdilated = cv2.dilate(fgthres, 
                                 kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                                 iterations=3)
            
            # Predict gesture
            gesture_text, confidence = self.predict(fgdilated)
            
            # Update gesture history
            if len(self.gesture_history) >= self.max_history_length:
                self.gesture_history.pop(0)
            self.gesture_history.append((gesture_text, confidence))
            
            # Execute gesture command if enabled
            if confidence > self.settings['gesture_sensitivity']:
                self.execute_gesture_command(gesture_text)
            
            # Record data if enabled
            if self.settings['recording']:
                self.recording_data.append({
                    'timestamp': time.time(),
                    'gesture': gesture_text,
                    'confidence': float(confidence)
                })
            
            # Update and draw interface
            self.update_fps()
            frame = self.draw_interface(frame, gesture_text, confidence)
            
            # Show debug information if enabled
            if self.settings['debug_mode']:
                cv2.imshow('Foreground Mask', fgmask)
                cv2.imshow('Processed Mask', fgdilated)
            
            # Display main frame
            cv2.imshow(self.windowName, frame)
            
            # Handle key presses
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.settings['muted'] = not self.settings['muted']
            elif key == ord('r'):
                self.settings['recording'] = not self.settings['recording']
                if not self.settings['recording']:
                    self.save_gesture_data()
            elif key == ord('d'):
                self.settings['debug_mode'] = not self.settings['debug_mode']
            elif key == ord('s'):
                self.settings['system_control'] = not self.settings['system_control']
            elif key == ord('+'):
                self.settings['gesture_sensitivity'] = min(1.0, 
                    self.settings['gesture_sensitivity'] + 0.1)
            elif key == ord('-'):
                self.settings['gesture_sensitivity'] = max(0.1, 
                    self.settings['gesture_sensitivity'] - 0.1)
        
        # Cleanup
        self.vc.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        if self.settings['recording']:
            self.save_gesture_data()

if __name__ == "__main__":
    app = GestureRecognitionSystem()
    app.run()
