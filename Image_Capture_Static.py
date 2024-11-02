import numpy as np
import cv2
import os
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Window names and setup
windows = {
    'main': "Live Camera Input",
    'background': "Background Model",
    'foreground': "Foreground Objects",
    'probability': "Foreground Probability"
}

# Create windows
for window in windows.values():
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

# Configuration
config = {
    'video_path': './gesture1.MOV',
    'output_path': './gesture/1',
    'total_frames': 1200,
    'frame_size': (50, 50),
    'skip_factor': 0.1  # 10% of video duration
}

# Create output directory if it doesn't exist
os.makedirs(config['output_path'], exist_ok=True)

# Load metadata
try:
    df = pd.read_excel('Videos_Intention.xlsx')
    filtered_df = df.loc[df['Left Forearm '] == 2]
    logger.info("Successfully loaded metadata")
except Exception as e:
    logger.error(f"Error loading metadata: {e}")

def calculate_frame_skip(video_capture):
    """Calculate number of frames to skip based on video properties"""
    video_capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    total_ms = video_capture.get(cv2.CAP_PROP_POS_MSEC)
    ms_to_skip = config['skip_factor'] * total_ms
    seconds_to_skip = 0.001 * ms_to_skip
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    return int(fps * seconds_to_skip)

def setup_video_capture():
    """Initialize video capture and calculate frame skip"""
    cap = cv2.VideoCapture(config['video_path'])
    if not cap.isOpened():
        raise Exception("Error opening video file")
    
    skip_frames = calculate_frame_skip(cap)
    logger.info(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    logger.info(f"Frames to skip: {skip_frames}")
    
    # Reset video capture
    cap.release()
    cap = cv2.VideoCapture(config['video_path'])
    return cap, skip_frames

def process_frame(frame, mog):
    """Process a single frame with background subtraction"""
    if frame is None:
        return None, None, None, None
        
    frame = cv2.flip(frame, 1)
    fgmask = mog.apply(frame)
    fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
    fgdilated = cv2.dilate(
        fgthres,
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
        iterations=3
    )
    bgmodel = mog.getBackgroundImage()
    
    return frame, fgmask, fgdilated, bgmodel

def save_frame(frame, frame_number):
    """Save processed frame"""
    try:
        save_img = cv2.resize(frame, config['frame_size'])
        save_img = np.array(save_img)
        filepath = os.path.join(config['output_path'], f"{frame_number}.jpg")
        cv2.imwrite(filepath, save_img)
        return True
    except Exception as e:
        logger.error(f"Error saving frame {frame_number}: {e}")
        return False

def main():
    try:
        # Initialize video capture and calculate frame skip
        vc, skip_frames = setup_video_capture()
        rval, frame = vc.read()
        
        # Initialize variables
        pic_no = 0
        frame_count = 0
        flag_capturing = True
        processing_start = datetime.now()
        
        # Create background subtractor
        mog = cv2.createBackgroundSubtractorMOG2(
            history=2000,
            varThreshold=16,
            detectShadows=True
        )
        
        logger.info("Starting frame extraction...")
        
        while True:
            if frame is not None:
                # Process frame
                frame, fgmask, fgdilated, bgmodel = process_frame(frame, mog)
                
                # Display frames
                cv2.imshow(windows['main'], frame)
                cv2.imshow(windows['foreground'], fgdilated)
                cv2.imshow(windows['probability'], fgmask)
                cv2.imshow(windows['background'], bgmodel)
                
                # Save frame if capturing
                if flag_capturing:
                    frame_count += 1
                    if frame_count >= skip_frames:
                        pic_no += 1
                        if save_frame(fgdilated, pic_no):
                            frame_count = 0
                            # Log progress every 100 frames
                            if pic_no % 100 == 0:
                                elapsed = datetime.now() - processing_start
                                rate = pic_no / elapsed.total_seconds()
                                logger.info(f"Processed {pic_no} frames ({rate:.2f} fps)")
                
                # Optional: Stop at total frames
                if pic_no == config['total_frames']:
                    logger.info(f"Reached target of {config['total_frames']} frames")
                    break
            
            # Read next frame
            rval, frame = vc.read()
            if not rval:
                logger.info("End of video reached")
                break
            
            # Handle key presses
            keypress = cv2.waitKey(1)
            if keypress == ord('q'):
                logger.info("Processing stopped by user")
                break
            elif keypress == ord('c'):
                flag_capturing = not flag_capturing
                logger.info(f"Capture {'enabled' if flag_capturing else 'disabled'}")
            elif keypress == ord('s'):
                # Save current frame regardless of skip count
                pic_no += 1
                save_frame(fgdilated, pic_no)
                logger.info(f"Manually saved frame {pic_no}")
                
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        
    finally:
        # Cleanup
        vc.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        # Log summary
        total_time = datetime.now() - processing_start
        logger.info(f"Processing complete. Saved {pic_no} frames in {total_time}")

if __name__ == "__main__":
    main()
