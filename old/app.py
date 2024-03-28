import whisper
import datetime
import soundfile as sf
from pyannote.audio import Pipeline

# Load the Pyannote speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")

path = "./audio/standardized.wav"

# Load your audio file
audio, rate = sf.read(path)

# Apply the diarization pipeline on the entire audio file
diarization = pipeline(path)

# Load the Whisper model for transcription
language = 'English'  # Set language
model_size = 'base'  # Choose Whisper model size: tiny, base, small, medium, large
whisper_model = whisper.load_model(model_size)

# Transcribe the entire audio file
result = whisper_model.transcribe(path)
segments = result["segments"]

# Function to format timedelta strings
def format_timedelta(td):
    # Strip microsecond component from timedelta string
    return str(td).split('.')[0]

transcript_path = "transcript.txt"
with open(transcript_path, "w") as f:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_tag = f"SPEAKER {speaker}"
        start_time = format_timedelta(datetime.timedelta(seconds=turn.start))
        end_time = format_timedelta(datetime.timedelta(seconds=turn.end))
        # Fetch transcript segment that overlaps with current speaker turn
        transcript_segment = " ".join(seg["text"] for seg in segments if seg["start"] >= turn.start and seg["end"] <= turn.end)
        # Write speaker tag and corresponding transcript to file
        f.write(f"{speaker_tag} {start_time} - {end_time}\n")
        f.write(f"{transcript_segment}\n\n")

# Print the transcript
with open(transcript_path, "r") as f:
    print(f.read())