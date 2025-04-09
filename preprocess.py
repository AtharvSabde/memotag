import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal

def preprocess_audio(input_folder, output_folder, min_pause_duration=0.5):
    """
    Preprocess audio files in the input folder and save to output folder.
    
    Parameters:
    -----------
    input_folder : str
        Path to folder containing MP3 audio files
    output_folder : str
        Path to save preprocessed files
    min_pause_duration : float
        Minimum duration (in seconds) to consider as a pause
    
    Returns:
    --------
    dict
        Dictionary with filenames as keys and lists of pause durations as values
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Dictionary to store pause durations for each file
    all_pause_durations = {}
    
    # Process each MP3 file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp3'):
            file_path = os.path.join(input_folder, filename)
            
            # Load audio file
            print(f"Processing {filename}...")
            y, sr = librosa.load(file_path, sr=None)
            
            # Step 1: Remove background noise using spectral gating
            y_filtered = remove_noise(y, sr)
            
            # Step 2: Remove leading/trailing silence
            y_trimmed, trim_indices = librosa.effects.trim(y_filtered, top_db=20)
            
            # Step 3: Detect pauses
            pause_durations = detect_pauses(y_trimmed, sr, min_pause_duration)
            all_pause_durations[filename] = pause_durations
            
            # Save preprocessed audio
            output_path = os.path.join(output_folder, f"processed_{filename}")
            sf.write(output_path, y_trimmed, sr)
            
            print(f"Saved preprocessed file to {output_path}")
            print(f"Found {len(pause_durations)} pauses")
            if pause_durations:
                print(f"Pause durations (seconds): {[round(d, 2) for d in pause_durations]}")
            print("------------------------")
    
    return all_pause_durations

def remove_noise(y, sr):
    """
    Remove background noise using spectral gating
    
    Parameters:
    -----------
    y : numpy.ndarray
        Audio time series
    sr : int
        Sample rate
    
    Returns:
    --------
    numpy.ndarray
        Noise-reduced audio time series
    """
    # Use a simpler approach with librosa's decompose function
    # This separates harmonic and percussive components, which can help reduce noise
    y_harmonic = librosa.effects.preemphasis(y)
    
    # Additional noise reduction with a high-pass filter to remove low-frequency noise
    # Define the filter parameters
    cutoff_freq = 80  # Hz - adjust based on your specific needs
    nyquist = sr / 2
    normal_cutoff = cutoff_freq / nyquist
    
    # Create a high-pass filter
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    
    # Apply the filter
    y_filtered = signal.filtfilt(b, a, y_harmonic)
    
    return y_filtered

def detect_pauses(y, sr, min_pause_duration=0.5):
    """
    Detect pauses in audio longer than min_pause_duration
    
    Parameters:
    -----------
    y : numpy.ndarray
        Audio time series
    sr : int
        Sample rate
    min_pause_duration : float
        Minimum duration (in seconds) to consider as a pause
    
    Returns:
    --------
    list
        List of pause durations in seconds
    """
    # Calculate energy envelope
    frame_length = int(0.025 * sr)  # 25 ms frames
    hop_length = int(0.010 * sr)    # 10 ms hop
    
    # RMS energy
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Normalize energy to 0-1
    energy_norm = energy / np.max(energy) if np.max(energy) > 0 else energy
    
    # Threshold for detecting silence (adjust as needed)
    silence_threshold = 0.05
    
    # Detect silent frames
    silent_frames = energy_norm < silence_threshold
    
    # Convert frames to time
    frame_time = hop_length / sr
    
    # Find continuous silent segments
    pause_durations = []
    in_pause = False
    pause_start = 0
    
    for i, is_silent in enumerate(silent_frames):
        if is_silent and not in_pause:
            # Start of a new pause
            in_pause = True
            pause_start = i
        elif not is_silent and in_pause:
            # End of a pause
            pause_duration = (i - pause_start) * frame_time
            if pause_duration >= min_pause_duration:
                pause_durations.append(pause_duration)
            in_pause = False
    
    # Check if we ended in a pause
    if in_pause:
        pause_duration = (len(silent_frames) - pause_start) * frame_time
        if pause_duration >= min_pause_duration:
            pause_durations.append(pause_duration)
    
    return pause_durations

if __name__ == "__main__":
    # Example usage
    input_folder = r"C:\Users\Admin\Desktop\memotag_test\samples"
    output_folder = r"C:\Users\Admin\Desktop\memotag_test\audio_folder"
    
    # You can change min_pause_duration as needed
    pause_data = preprocess_audio(input_folder, output_folder, min_pause_duration=0.5)
    
    # Print summary
    print("\nPreprocessing Summary:")
    print("====================")
    for filename, pauses in pause_data.items():
        avg_pause = sum(pauses) / len(pauses) if pauses else 0
        print(f"{filename}: {len(pauses)} pauses, avg duration: {round(avg_pause, 2)}s")