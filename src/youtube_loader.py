import re
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document

# Setup logging
logger = logging.getLogger(__name__)

def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character video ID from any standard YouTube link.
    e.g. https://www.youtube.com/watch?v=dQw4w9WgXcQ -> dQw4w9WgXcQ
         https://youtu.be/dQw4w9WgXcQ -> dQw4w9WgXcQ
    """
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url.strip())
    if match:
        return match.group(1)
    return None

def load_youtube_transcript(url: str) -> list[Document]:
    """
    Fetches the subtitles/transcript from a YouTube video and aggregates them into a Document.
    Supports both standard class-method API (get_transcript) and instance-method API (fetch).
    
    Args:
        url: The YouTube video URL.
        
    Returns:
        A list containing one LangChain Document containing the transcript text.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL. Please provide a standard YouTube video link.")
        
    logger.info(f"Retrieving YouTube transcript for video ID: {video_id}...")
    try:
        # Check API type dynamically
        if hasattr(YouTubeTranscriptApi, 'get_transcript'):
            # Standard classmethod-based API
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['en', 'en-US', 'hi']
            )
            segments = transcript_list
        else:
            # Fallback instance-based API (found in older/custom versions like 1.2.4)
            api = YouTubeTranscriptApi()
            fetched_transcript = api.fetch(
                video_id, 
                languages=('en', 'en-US', 'hi')
            )
            segments = fetched_transcript.snippets
            
        # Structure transcript segments along with their timestamps
        full_text_segments = []
        for entry in segments:
            # Extract start time and text, supporting both dict and object structures
            if isinstance(entry, dict):
                start_time = int(entry.get('start', 0))
                text = entry.get('text', '')
            else:
                start_time = int(getattr(entry, 'start', 0))
                text = getattr(entry, 'text', '')
                
            minutes = start_time // 60
            seconds = start_time % 60
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            
            # Format text line
            full_text_segments.append(f"{timestamp} {text}")
            
        full_text = "\n".join(full_text_segments)
        
        # Meta info
        metadata = {
            "source": "youtube",
            "video_id": video_id,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "file_name": f"youtube_{video_id}.txt",
            "file_path": f"youtube_{video_id}",
            "language": "Transcript"
        }
        
        doc = Document(
            page_content=full_text,
            metadata=metadata
        )
        logger.info("Successfully fetched transcript.")
        return [doc]
        
    except Exception as e:
        logger.error(f"Failed to fetch YouTube transcript: {e}")
        raise RuntimeError(
            f"Could not retrieve transcript for the video. Please verify the URL, "
            f"and make sure the video has public captions/subtitles enabled. Error: {str(e)}"
        )
