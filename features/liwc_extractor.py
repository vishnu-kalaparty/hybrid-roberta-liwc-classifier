"""
LIWC-like Feature Extractor
============================
Extracts LIWC-like linguistic features from text.
Since LIWC is proprietary, we create similar lexicon-based features.
"""

import re
import numpy as np


class LIWCFeatureExtractor:
    """
    Extracts LIWC-like linguistic features from text.
    Since LIWC is proprietary, we create similar lexicon-based features.
    """
    
    def __init__(self):
        # Define lexicons for various categories
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'happy', 'joy', 'love', 'excited', 'pleased', 'grateful', 'thankful',
            'successful', 'achieve', 'accomplish', 'perfect', 'best', 'brilliant',
            'impressive', 'outstanding', 'remarkable', 'superb', 'delighted'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'hate',
            'angry', 'sad', 'upset', 'disappointed', 'frustrated', 'annoyed',
            'fail', 'failure', 'wrong', 'mistake', 'problem', 'issue', 'concern',
            'worry', 'anxious', 'stressed', 'tired', 'exhausted', 'boring'
        }
        
        self.confident_words = {
            'certainly', 'definitely', 'absolutely', 'clearly', 'obviously',
            'undoubtedly', 'surely', 'indeed', 'demonstrate', 'prove', 'confirm',
            'establish', 'show', 'evidence', 'significant', 'substantial'
        }
        
        self.tentative_words = {
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'may', 'seem',
            'appear', 'suggest', 'assume', 'guess', 'think', 'believe',
            'suppose', 'probably', 'likely', 'unlikely', 'unsure', 'uncertain'
        }
        
        self.anxious_words = {
            'worried', 'nervous', 'anxious', 'afraid', 'scared', 'fear',
            'panic', 'stress', 'tense', 'uneasy', 'concern', 'doubt',
            'uncertain', 'hesitant', 'overwhelmed', 'desperate'
        }
        
        self.frustrated_words = {
            'frustrated', 'annoyed', 'irritated', 'fed up', 'exasperated',
            'impatient', 'aggravated', 'bothered', 'tired of', 'sick of',
            'enough', 'stop', 'quit', 'give up', 'hopeless'
        }
        
        # Formality markers
        self.informal_markers = {
            'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'yeah', 'yep', 'nope',
            'ok', 'okay', 'cool', 'awesome', 'stuff', 'things', 'guy', 'guys',
            'lol', 'omg', 'btw', 'idk', 'tbh', 'imo', 'u', 'ur', 'r', 'y',
            'cuz', 'cos', 'cause', 'tho', 'tho'
        }
        
        self.formal_markers = {
            'therefore', 'consequently', 'furthermore', 'moreover', 'however',
            'nevertheless', 'nonetheless', 'accordingly', 'hence', 'thus',
            'regarding', 'concerning', 'pursuant', 'whereby', 'whereas',
            'demonstrate', 'indicate', 'suggest', 'propose', 'establish'
        }
        
        # Pronouns
        self.first_person = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'}
        self.second_person = {'you', 'your', 'yours', 'yourself', 'yourselves'}
        self.third_person = {'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their', 'theirs'}
        
    def extract_features(self, text):
        """Extract all lexicon-based features from text."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = max(len(words), 1)  # Avoid division by zero
        
        features = {}
        
        # Emotion-related features (normalized by word count)
        features['positive_ratio'] = len([w for w in words if w in self.positive_words]) / word_count
        features['negative_ratio'] = len([w for w in words if w in self.negative_words]) / word_count
        features['confident_ratio'] = len([w for w in words if w in self.confident_words]) / word_count
        features['tentative_ratio'] = len([w for w in words if w in self.tentative_words]) / word_count
        features['anxious_ratio'] = len([w for w in words if w in self.anxious_words]) / word_count
        features['frustrated_ratio'] = len([w for w in words if w in self.frustrated_words]) / word_count
        
        # Formality features
        features['informal_ratio'] = len([w for w in words if w in self.informal_markers]) / word_count
        features['formal_ratio'] = len([w for w in words if w in self.formal_markers]) / word_count
        
        # Pronoun features
        features['first_person_ratio'] = len([w for w in words if w in self.first_person]) / word_count
        features['second_person_ratio'] = len([w for w in words if w in self.second_person]) / word_count
        features['third_person_ratio'] = len([w for w in words if w in self.third_person]) / word_count
        
        # Structural features
        features['exclamation_count'] = text.count('!') / word_count
        features['question_count'] = text.count('?') / word_count
        features['contraction_count'] = len(re.findall(r"\b\w+'\w+\b", text_lower)) / word_count
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_length'] = word_count
        
        return list(features.values())
    
    def get_feature_names(self):
        """Return feature names for interpretability."""
        return [
            'positive_ratio', 'negative_ratio', 'confident_ratio', 'tentative_ratio',
            'anxious_ratio', 'frustrated_ratio', 'informal_ratio', 'formal_ratio',
            'first_person_ratio', 'second_person_ratio', 'third_person_ratio',
            'exclamation_count', 'question_count', 'contraction_count',
            'caps_ratio', 'avg_word_length', 'sentence_length'
        ]
