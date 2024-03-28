from pydub import AudioSegment
import noisereduce as nr
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.io import wavfile
import os
import torch
import whisper
from pyannote.audio import Pipeline, Inference
from pyannote.core import Segment
from typing import List, Tuple
from pyannote.audio.pipelines.utils.hook import ProgressHook

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def convert_and_preprocess(input_file_path, output_file_path='./audio/standardized.wav'):
    # Convert audio file to wav format
    audio = AudioSegment.from_file(input_file_path)
    audio = audio.set_frame_rate(16000)  # Set the frame rate to 16 kHz
    audio = audio.set_channels(1)  # Ensure audio is mono

    # Normalize audio to a target amplitude
    target_dBFS = -20.0
    change_in_dBFS = target_dBFS - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)

    # Export the normalized audio to a wav file
    audio.export(output_file_path, format='wav')
    
    # Read the wav file data for noise reduction processing
    sample_rate, data = wavfile.read(output_file_path)
    data = data.astype(float)

    # Apply noise reduction
    reduced_noise_data = nr.reduce_noise(y=data, sr=sample_rate)
    
    # Convert the float audio data to int16, as wav files use 16-bit PCM
    reduced_noise_data = reduced_noise_data.astype(np.int16)
    
    # Save the noise-reduced audio data back to a wav file
    wavfile.write(output_file_path, sample_rate, reduced_noise_data)

def diarize_speakers() -> List[Tuple[float, float, str]]:
    # Load the speaker diarization pipeline (using a pre-trained model)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
    
    with ProgressHook() as hook:
      # Process the audio file
      diarization = pipeline("./audio/standardized.wav", hook=hook)
    
    # Initialize a list to hold the output segments
    segments = []

    # Iterate over the diarization result to prepare the output
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end
        # Append the segment information to the segments list
        segments.append((start_time, end_time, speaker))
    
    # Sort the segments list by start time
    segments.sort(key=lambda x: x[0])
    
    return segments

def extract_speaker_embeddings(speaking_segments, model_inference, audio_path):
    embeddings = []
    
    for segment in speaking_segments:
        # Define the segment using pyannote.core.Segment
        start_time, end_time, _ = segment
        segment = Segment(start=start_time, end=end_time)
        
        # Extract the embedding for the defined segment
        embedding = model_inference.crop(audio_path, segment)
        embeddings.append(embedding)
        
    return embeddings

def perform_asr():
  # Load the Whisper model
  model = whisper.load_model("large")  # Choose between "tiny", "base", "small", "medium", "large" based on your needs and resources

  # Transcribe the audio file with timestamps
  audio_file_path = "./audio/standardized.wav"  # Replace with the path to your actual file
  result_segments = model.transcribe(audio_file_path)

  # The result contains a 'segments' key with the transcription and timestamps for each segment
  for segment in result_segments["segments"]:
      print(f"{segment['start']}s - {segment['end']}s: {segment['text']}")

  return result_segments

# Replace 'input_file_path' with the path to your audio file
convert_and_preprocess('./audio/input.mp3')

speaking_segments = diarize_speakers()
# Load the speaker embedding model
embedding_model = Inference("pyannote/embedding", use_auth_token="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")

# Assuming 'speaking_segments' contains the results from your diarization
speaker_embeddings = extract_speaker_embeddings(speaking_segments, embedding_model, "./audio/standardized.wav")

# Flatten the list of SlidingWindowFeature objects into a matrix for clustering
embedding_matrix = np.vstack([embedding.data for embedding in speaker_embeddings])

# Use agglomerative clustering to cluster the embeddings matrix
# Since you know there are 2 speakers, you can use n_clusters=2
cluster_model = AgglomerativeClustering(n_clusters=2)
labels = cluster_model.fit_predict(embedding_matrix)


# Use t-SNE to reduce the dimensionality of the embeddings to 2D for visualization
tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(embedding_matrix)

# Plot the reduced embeddings with different colors for different clusters
plt.figure(figsize=(8, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
plt.title('2D Visualization of Speaker Embeddings')
plt.legend(handles=scatter.legend_elements()[0], labels=set(labels))
plt.grid(True)
plt.show()

for segment in speaking_segments:
    print(f"Speaker {segment[2]} from {segment[0]:.2f} to {segment[1]:.2f} seconds.")

# text_segments = perform_asr()