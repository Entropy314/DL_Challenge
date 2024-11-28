import re
import pretty_midi
import numpy as np
import pandas as pd






def extract_basic_features(midi):
    """
    Extracts a comprehensive set of features from a MIDI file.
    
    Parameters:
    - file_path: Path to the MIDI file.

    Returns:
    - Dictionary of extracted features.
    """
    
    pitches, durations, velocities, tempos, instruments = [], [], [], [], []
    
    # Extract note-level features
    for instrument in midi.instruments:
        if not instrument.is_drum:  # Exclude drums
            instruments.append(instrument.program)
            for note in instrument.notes:
                pitches.append(note.pitch)
                durations.append(note.end - note.start)
                velocities.append(note.velocity)

    # Tempo changes
    tempo_changes = midi.get_tempo_changes()[0]
    tempos = tempo_changes if tempo_changes.size > 0 else [120]  # Default to 120
    filename = filepath.split('/')[-1].replace(' in ', ' ')

    
    # Feature calculations
    features = {
        # Pitch Features
        "mean_pitch": np.mean(pitches) if pitches else np.nan,
        "std_pitch": np.std(pitches) if pitches else np.nan,
        "pitch_range": (max(pitches) - min(pitches)) if pitches else np.nan,
        "pitch_min": min(pitches), "pitch_max": max(pitches), 
        "pitch_class_entropy": calculate_entropy([p % 12 for p in pitches]),

        # Duration Features
        "mean_duration": np.mean(durations) if durations else np.nan,
        "std_duration": np.std(durations) if durations else np.nan,
        "note_density": len(durations) / midi.get_end_time() if durations else np.nan,

        # Tempo Features
        "mean_tempo": np.mean(tempos),
        "tempo_std": np.std(tempos),

        # Dynamics Features
        "mean_velocity": np.mean(velocities) if velocities else np.nan,
        "velocity_range": (max(velocities) - min(velocities)) if velocities else np.nan,

        # Instrumentation Features
        "num_instruments": len(set(instruments)),
        "instrument_entropy": calculate_entropy(instruments),

        
    }
    return features

def calculate_entropy(elements):
    """
    Calculates entropy for a list of elements.

    Parameters:
    - elements: List of elements (e.g., pitch classes or instruments).

    Returns:
    - Entropy value.
    """
    if not elements:
        return 0
    counts = np.bincount(elements)
    probabilities = counts / len(elements)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


def extract_chord_features(midi):
    """
    Extracts harmony-related features from a MIDI object.
    
    Parameters:
    - midi: A pretty_midi.PrettyMIDI object.

    Returns:
    - Dictionary of chord-related features.
    """
    chords = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes = [note.pitch for note in instrument.notes]
            chords.extend([(n1 % 12, n2 % 12) for n1, n2 in zip(notes[:-1], notes[1:])])
    
    unique_chords = len(set(chords))
    return {
        "chord_diversity": unique_chords,
        "most_common_chord": max(set(chords), key=chords.count) if chords else None,
    }


# 1. Load and extract features from MIDI files
def extract_all_features(file_path):
    """
    Extracts all features from a MIDI file and combines them into a single dictionary.

    Parameters:
    - file_path: Path to the MIDI file.

    Returns:
    - Dictionary of all extracted features.
    """
    midi = pretty_midi.PrettyMIDI(file_path)
    basic_features = extract_basic_features(midi)
    chord_features = extract_chord_features(midi)
    return {**basic_features, **chord_features}


def extract_basic_features_for_chunk(midi, start_time, end_time):
    """
    Extracts features from a specific time chunk of a MIDI file.

    Parameters:
    - midi: A pretty_midi.PrettyMIDI object.
    - start_time: Start time of the chunk in seconds.
    - end_time: End time of the chunk in seconds.

    Returns:
    - Dictionary of extracted features for the time chunk.
    """
    pitches, durations, velocities, tempos, instruments = [], [], [], [], []

    # Extract note-level features within the time chunk
    for instrument in midi.instruments:
        if not instrument.is_drum:  # Exclude drums
            instruments.append(instrument.program)
            for note in instrument.notes:
                if start_time <= note.start < end_time:
                    pitches.append(note.pitch)
                    durations.append(note.end - note.start)
                    velocities.append(note.velocity)

    # Tempo changes within the time chunk
    tempo_changes, tempo_times = midi.get_tempo_changes()
    relevant_tempos = [tempo for tempo, time in zip(tempo_changes, tempo_times)
                       if start_time <= time < end_time]
    tempos = relevant_tempos if relevant_tempos else [120]  # Default to 120 if no changes

    # Feature calculations
    features = {
        # Pitch Features
        "mean_pitch": np.mean(pitches) if pitches else np.nan,
        "std_pitch": np.std(pitches) if pitches else np.nan,
        "pitch_range": (max(pitches) - min(pitches)) if pitches else np.nan,
        "pitch_min": min(pitches) if pitches else np.nan,
        "pitch_max": max(pitches) if pitches else np.nan,
        "pitch_class_entropy": calculate_entropy([p % 12 for p in pitches]),

        # Duration Features
        "mean_duration": np.mean(durations) if durations else np.nan,
        "std_duration": np.std(durations) if durations else np.nan,
        "note_density": len(durations) / (end_time - start_time) if durations else np.nan,

        # Tempo Features
        "mean_tempo": np.mean(tempos),
        "tempo_std": np.std(tempos),

        # Dynamics Features
        "mean_velocity": np.mean(velocities) if velocities else np.nan,
        "velocity_range": (max(velocities) - min(velocities)) if velocities else np.nan,

        # Instrumentation Features
        "num_instruments": len(set(instruments)),
        "instrument_entropy": calculate_entropy(instruments),
    }
    return features


def extract_chord_features_for_chunk(midi, start_time, end_time):
    """
    Extracts chord-related features from a specific time chunk of a MIDI file.

    Parameters:
    - midi: A pretty_midi.PrettyMIDI object.
    - start_time: Start time of the chunk in seconds.
    - end_time: End time of the chunk in seconds.

    Returns:
    - Dictionary of chord-related features for the time chunk.
    """
    chords = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes = [note.pitch for note in instrument.notes
                     if start_time <= note.start < end_time]
            chords.extend([(n1 % 12, n2 % 12) for n1, n2 in zip(notes[:-1], notes[1:])])

    unique_chords = len(set(chords))
    return {
        "chord_diversity": unique_chords,
        "most_common_chord": max(set(chords), key=chords.count) if chords else None,
    }

def extract_fourier_features(note_sequence, sample_rate=1):
    """
    Computes Fourier transform features from a note sequence.

    Parameters:
    - note_sequence: List or array of numeric values (e.g., pitches, velocities).
    - sample_rate: Sampling rate of the note sequence (default: 1).

    Returns:
    - Dictionary of Fourier transform features.
    """
    if len(note_sequence) == 0:
        # Handle empty input
        return {
            "dominant_frequency": 0,
            "spectral_centroid": 0,
            "spectral_entropy": 0,
            "spectral_flatness": 0,
        }

    # Compute the Fourier Transform
    spectrum = np.fft.fft(note_sequence)
    magnitudes = np.abs(spectrum)
    frequencies = np.fft.fftfreq(len(note_sequence), d=1/sample_rate)

    # Focus on positive frequencies
    positive_freqs = frequencies[:len(frequencies)//2]
    positive_magnitudes = magnitudes[:len(magnitudes)//2]

    # Features
    dominant_frequency = positive_freqs[np.argmax(positive_magnitudes)]
    spectral_centroid = np.sum(positive_freqs * positive_magnitudes) / np.sum(positive_magnitudes)
    spectral_entropy = -np.sum((positive_magnitudes / np.sum(positive_magnitudes)) * 
                               np.log2(positive_magnitudes / np.sum(positive_magnitudes) + 1e-10))
    spectral_flatness = np.exp(np.mean(np.log(positive_magnitudes + 1e-10))) / np.mean(positive_magnitudes)

    return {
        "dominant_frequency": dominant_frequency,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
        "spectral_flatness": spectral_flatness,
    }


def process_midi_in_chunks(file_path:str, chunk_size:int= 30, stride:int=10, sample_rate:int=1):
    """
    Process a MIDI file and extract features for 30-second chunks.

    Parameters:
    - file_path: Path to the MIDI file.
    - chunk_size: Length of each time chunk in seconds.
    - stride: Step size to shift the chunk window.

    Returns:
    - List of dictionaries with features for each chunk.
    """
    # Load the MIDI file
    midi = pretty_midi.PrettyMIDI(file_path)
    total_duration = midi.get_end_time()

    chunk_features = []
    composer = file_path.split('/')[-2]

    # Process each chunk
    start_time = 0
    while start_time + chunk_size <= total_duration:
        end_time = start_time + chunk_size

        # Extract basic and chord features
        basic_features = extract_basic_features_for_chunk(midi, start_time, end_time)
        chord_features = extract_chord_features_for_chunk(midi, start_time, end_time)

        # Extract Fourier features
        pitches = [
            note.pitch for instrument in midi.instruments if not instrument.is_drum
            for note in instrument.notes if start_time <= note.start < end_time
        ]
        # print(pitches)
        fourier_features = extract_fourier_features(pitches, sample_rate=len(pitches)/chunk_size)
        # Combine features
        combined_features = {
            **basic_features,
            **chord_features,
            **fourier_features,
            "start_time": start_time,
            "end_time": end_time,
            "composer": composer,
        }
        chunk_features.append(combined_features)

        # Increment start_time by the stride
        start_time += stride

    return chunk_features



def extract_metadata(filename:str) -> dict: 

    pattern = r"(?P<title>.+?)_(?P<catalog>.+?)_(?P<id>\d+?)_(?P<code>.+?)"


    # Extract metadata
    match = re.match(pattern, filename)
    result = match.groupdict()
    title = result['title'].replace(' in ', ' ')
    if 'minor' in title: 
        result['meta_scale'] = 'minor'
    elif 'major' in title: 
        result['meta_scale'] = 'major'
    else: 
        result['scale'] = 'NA'
    if 'No' in title: 
        result['meta_number'] = title.split(' ')[3]
    else: 
        result['meta_number'] = 0 
    
    title_array = title.split(' ')
    result['meta_instrument'] = title_array[0]
    result['meta_0'] = title_array[1]
    return result


