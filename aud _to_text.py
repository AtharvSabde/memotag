import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def transcribe_audio_files(input_folder, output_folder, model_size="base"):
    """
    Transcribe audio files using Hugging Face's Whisper implementation.
    
    Parameters:
    -----------
    input_folder : str
        Path to folder containing processed audio files
    output_folder : str
        Path to save transcriptions
    model_size : str
        Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    
    Returns:
    --------
    dict
        Dictionary with filenames as keys and transcriptions as values
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Whisper model from Hugging Face
    model_id = f"openai/whisper-{model_size}"
    print(f"Loading Whisper model {model_id}...")
    
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    
    # Dictionary to store transcriptions
    all_transcriptions = {}
    
    # Transcribe each audio file in the input folder
    for filename in tqdm(os.listdir(input_folder), desc="Transcribing files"):
        if filename.endswith(('.mp3', '.wav')):
            file_path = os.path.join(input_folder, filename)
            
            # Transcribe audio
            print(f"\nTranscribing {filename}...")
            
            # Use librosa to load audio (more flexible with various formats)
            import librosa
            audio, sample_rate = librosa.load(file_path, sr=16000)
            
            # Process audio with Whisper
            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            
            # Generate token ids
            predicted_ids = model.generate(input_features)
            
            # Decode token ids to text
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Store transcription
            all_transcriptions[filename] = transcription
    
    # Create a consolidated file with all transcriptions
    consolidated_path = os.path.join(output_folder, "all_transcriptions.txt")
    with open(consolidated_path, 'w', encoding='utf-8') as f:
        for filename, transcription in all_transcriptions.items():
            f.write(f"File: {filename}\n")
            f.write(f"Transcription: {transcription}\n")
            f.write("-" * 80 + "\n")  # Separator between entries
    
    print(f"\nAll transcriptions saved to single file: {consolidated_path}")
    
    # Also keep the CSV output for convenience
    df = pd.DataFrame({
        'filename': list(all_transcriptions.keys()),
        'transcription': list(all_transcriptions.values())
    })
    csv_path = os.path.join(output_folder, "all_transcriptions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Consolidated transcriptions also saved to CSV: {csv_path}")
    
    return all_transcriptions

if __name__ == "__main__":
    
    processed_audio_folder = r"C:\Users\Admin\Desktop\memotag_test\audio_folder"
    transcriptions_output_folder = r"C:\Users\Admin\Desktop\memotag_test\transcibe"
    
    
    # Larger models are more accurate but require more computational resources
    transcriptions = transcribe_audio_files(processed_audio_folder, transcriptions_output_folder, model_size="base")