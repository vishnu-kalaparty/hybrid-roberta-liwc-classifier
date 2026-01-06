"""
Data loading and preprocessing functions.
"""

import os
import pandas as pd


def load_goemotions_data(data_dir):
    """Load and preprocess GoEmotions dataset."""
    print("Loading GoEmotions dataset...")
    
    # Load all three files
    dfs = []
    for i in range(1, 4):
        filepath = os.path.join(data_dir, 'GoEmotions', f'goemotions_{i}.csv')
        df = pd.read_csv(filepath)
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Drop duplicates based on text
    data = data.drop_duplicates(subset=['text'])
    
    # Emotion columns (from column index 9 onwards based on the dataset structure)
    emotion_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                    'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    
    # Map emotions to 6 categories - EXACTLY as per your specification
    emotion_mapping = {
        # Positive (11 emotions)
        'admiration': 'Positive', 
        'amusement': 'Positive', 
        'approval': 'Positive',
        'caring': 'Positive', 
        'excitement': 'Positive',
        'gratitude': 'Positive', 
        'joy': 'Positive', 
        'love': 'Positive',
        'optimism': 'Positive', 
        'pride': 'Positive', 
        'relief': 'Positive',
        
        # Frustrated (3 emotions)
        'anger': 'Frustrated',
        'annoyance': 'Frustrated',
        'disappointment': 'Frustrated',
        
        # Negative (5 emotions)
        'disapproval': 'Negative', 
        'disgust': 'Negative', 
        'grief': 'Negative',
        'remorse': 'Negative', 
        'sadness': 'Negative',
        
        # Confident (1 emotion)
        'realization': 'Confident',
        
        # Anxious (2 emotions)
        'fear': 'Anxious', 
        'nervousness': 'Anxious',
        
        # Neutral (6 emotions)
        'neutral': 'Neutral', 
        'confusion': 'Neutral',
        'curiosity': 'Neutral', 
        'desire': 'Neutral',
        'embarrassment': 'Neutral',
        'surprise': 'Neutral',
    }
    
    # Priority order: check specific emotions first, neutral last
    priority_order = [
        'fear', 'nervousness',  # Anxious
        'anger', 'annoyance', 'disappointment',  # Frustrated
        'disapproval', 'disgust', 'grief', 'remorse', 'sadness',  # Negative
        'realization',  # Confident
        'admiration', 'amusement', 'approval', 'caring', 'excitement',
        'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief',  # Positive
        'neutral', 'confusion', 'curiosity', 'desire', 'embarrassment', 'surprise'  # Neutral last
    ]
    
    # Determine the primary emotion for each row using the mapping
    def get_primary_emotion(row):
        for emotion in priority_order:
            if emotion in row and row[emotion] == 1:
                return emotion_mapping.get(emotion, 'Neutral')
        return 'Neutral'
    
    # Check which emotion columns exist
    existing_cols = [col for col in emotion_cols if col in data.columns]
    print(f"Found {len(existing_cols)} emotion columns in dataset")
    
    data['emotion_label'] = data.apply(get_primary_emotion, axis=1)
    
    # Create label encoding
    label_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Confident': 3, 'Frustrated': 4, 'Anxious': 5}
    data['emotion_encoded'] = data['emotion_label'].map(label_map)
    
    print(f"GoEmotions loaded: {len(data)} samples")
    print(f"Emotion distribution:\n{data['emotion_label'].value_counts()}")
    
    return data[['text', 'emotion_label', 'emotion_encoded']]


def load_formality_data(data_dir):
    """Load and preprocess Formality dataset."""
    print("\nLoading Formality dataset...")
    
    filepath = os.path.join(data_dir, 'FormalityDataset.csv')
    data = pd.read_csv(filepath)
    
    # The dataset has 'text' and 'formality' columns
    # formality: 1 = Formal, 0 = Informal
    data = data[['text', 'formality']].dropna()
    data['formality_label'] = data['formality'].apply(lambda x: 'Formal' if x == 1 else 'Informal')
    data['formality_encoded'] = data['formality'].astype(int)
    
    print(f"Formality dataset loaded: {len(data)} samples")
    print(f"Formality distribution:\n{data['formality_label'].value_counts()}")
    
    return data[['text', 'formality_label', 'formality_encoded']]
