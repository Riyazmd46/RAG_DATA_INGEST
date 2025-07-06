import cv2
import os
from typing import List, Dict, Any
from PIL import Image
import numpy as np
from utils.text_utils import TextProcessor
from config import Config

class ImageProcessor:
    def __init__(self, api_key: str = None):
        self.text_processor = TextProcessor(api_key)
        
    def extract_video_frames(self, video_path: str, fps: int = None) -> List[Dict[str, Any]]:
        """Extract frames from video at specified FPS (default 1fps)."""
        fps = fps or Config.VIDEO_FPS
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {video_path}")
                return frames
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / video_fps
            
            # Calculate frame interval
            frame_interval = int(video_fps / fps)
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Save frame as temporary image
                    temp_path = f"temp_frame_{extracted_count}.jpg"
                    cv2.imwrite(temp_path, frame)
                    
                    # Generate description for the frame
                    description = self.text_processor.generate_image_description(temp_path)
                    
                    frame_data = {
                        'frame_number': frame_count,
                        'timestamp': frame_count / video_fps,
                        'description': description,
                        'image_path': temp_path
                    }
                    frames.append(frame_data)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            # Clean up temporary files
            for frame_data in frames:
                if os.path.exists(frame_data['image_path']):
                    os.remove(frame_data['image_path'])
            
            return frames
            
        except Exception as e:
            print(f"Error extracting frames from video {video_path}: {e}")
            return frames
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single image and generate description."""
        try:
            description = self.text_processor.generate_image_description(image_path)
            
            return {
                'image_path': image_path,
                'description': description,
                'file_size': os.path.getsize(image_path),
                'image_format': os.path.splitext(image_path)[1].lower()
            }
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return {
                'image_path': image_path,
                'description': "Image processing failed",
                'error': str(e)
            }
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video and extract frames with descriptions."""
        try:
            frames = self.extract_video_frames(video_path)
            
            # Combine all frame descriptions
            all_descriptions = [frame['description'] for frame in frames]
            combined_description = " ".join(all_descriptions)
            
            return {
                'video_path': video_path,
                'frames': frames,
                'total_frames_extracted': len(frames),
                'combined_description': combined_description,
                'file_size': os.path.getsize(video_path),
                'video_format': os.path.splitext(video_path)[1].lower()
            }
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return {
                'video_path': video_path,
                'error': str(e)
            } 