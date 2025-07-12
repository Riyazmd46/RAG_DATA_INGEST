import os
import whisper
from typing import Dict, Any, Optional
from pydub import AudioSegment
# import tempfile
from config import Config

class AudioProcessor:
    def __init__(self):
        self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
        
    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio from video file and return path to audio file."""
        try:
            # Create temporary file for audio
            temp_audio_path = os.path.splitext(video_path)[0] + "_extracted.wav"
            
            # Extract audio using pydub
            video = AudioSegment.from_file(video_path)
            audio = video.set_channels(1).set_frame_rate(16000)  # Convert to mono, 16kHz
            audio.export(temp_audio_path, format="wav")
            
            return temp_audio_path
        except Exception as e:
            print(f"Error extracting audio from video {video_path}: {e}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio file using Whisper."""
        try:
            # Load and transcribe audio
            result = self.whisper_model.transcribe(audio_path)
            
            return {
                'transcript': result['text'],
                'segments': result.get('segments', []),
                'language': result.get('language', 'unknown'),
                'audio_path': audio_path
            }
        except Exception as e:
            print(f"Error transcribing audio {audio_path}: {e}")
            return {
                'transcript': "Transcription failed",
                'error': str(e),
                'audio_path': audio_path
            }
    
    def process_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Process audio file and return transcription."""
        try:
            transcription = self.transcribe_audio(audio_path)
            
            return {
                'audio_path': audio_path,
                'transcription': transcription,
                'file_size': os.path.getsize(audio_path),
                'audio_format': os.path.splitext(audio_path)[1].lower()
            }
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")
            return {
                'audio_path': audio_path,
                'error': str(e)
            }
    
    def process_video_audio(self, video_path: str) -> Dict[str, Any]:
        """Extract audio from video and transcribe it."""
        try:
            # Extract audio from video
            audio_path = self.extract_audio_from_video(video_path)
            if not audio_path:
                return {
                    'video_path': video_path,
                    'error': 'Failed to extract audio from video'
                }
            
            # Transcribe the extracted audio
            transcription = self.transcribe_audio(audio_path)
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return {
                'video_path': video_path,
                'audio_transcription': transcription,
                'file_size': os.path.getsize(video_path),
                'video_format': os.path.splitext(video_path)[1].lower()
            }
        except Exception as e:
            print(f"Error processing video audio {video_path}: {e}")
            return {
                'video_path': video_path,
                'error': str(e)
            } 