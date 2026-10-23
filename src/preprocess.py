import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Setup NLTK downloads automatically
def setup_nltk():
    resources = {
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
        'corpora/omw-1.4': 'omw-1.4',
        'tokenizers/punkt': 'punkt',
        'sentiment/vader_lexicon': 'vader_lexicon'
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {name}...")
            try:
                nltk.download(name, quiet=True)
            except Exception as download_error:
                print(f"Could not download NLTK resource {name}: {download_error}")

# Run download check on import
setup_nltk()

# Initialize tools
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))

# Custom neutral filler words (pronouns, filler words)
custom_stops = {
    'im', 'ive', 'id', 'ill', 'youre', 'theyre', 'weve', 'were',
    'get', 'go', 'going', 'got', 'one', 'day', 'today', 'back', 'time', 'would', 'really',
    'still', 'well', 'think', 'see', 'know', 'amp', 'u', 'ur', 'us', 'much', 'many', 'make', 'take',
    'gonna', 'gotta', 'lol', 'lmao', 'oh', 'ok', 'okay', 'yeah', 'yes', 'thing', 'things'
}
_stop_words.update(custom_stops)

# CRITICAL: Negations & sentiment request words MUST NEVER be removed!
negation_and_sentiment_words = {
    'not', 'no', 'never', 'nor', 'neither', 'none', 'nobody', 'nothing', 'nowhere',
    'cannot', 'cant', 'dont', 'didnt', 'wont', 'isnt', 'arent', 'wasnt', 'werent',
    'hasnt', 'hadnt', 'havent', 'shouldnt', 'wouldnt', 'couldnt', 'without', 'lack',
    'hug', 'need', 'want', 'miss', 'hope', 'wish', 'love', 'hate', 'fix', 'save', 'kill', 'genocide'
}
_stop_words = _stop_words - negation_and_sentiment_words

def clean_tweet(text):
    """
    Cleans a tweet according to advanced preprocessing guidelines:
    - Converts to lowercase
    - Removes URLs, mentions (@username), hashtags (#happy -> happy)
    - Removes HTML tags and entities
    - Removes numbers, punctuation, emojis, and non-alphabetic characters
    - Fast tokenization, stopword filtering, and lemmatization
    """
    if not isinstance(text, str):
        return ""
        
    # 1. HTML tags and entities
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 4. Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 5. Extract hashtag words (#happy -> happy)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 6. Remove numbers, punctuation, emojis (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 7. Fast Tokenization & Stopword Removal
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        if token not in _stop_words and len(token) > 1:
            lemma = _lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemma)
            
    return " ".join(cleaned_tokens)

if __name__ == "__main__":
    # Test preprocessing
    test_tweets = [
        "I love this product! Check it out at https://example.com #awesome",
        "Hello @user, this is &amp; amazing 1234. 😍 I feel so happy!",
        "HTML <p>tags</p> should be removed and also stopwords like the, a, an."
    ]
    for tweet in test_tweets:
        # Encode-safe printing for Windows terminals
        safe_original = tweet.encode('ascii', errors='replace').decode('ascii')
        safe_cleaned = clean_tweet(tweet).encode('ascii', errors='replace').decode('ascii')
        print(f"Original: {safe_original}")
        print(f"Cleaned:  {safe_cleaned}")
        print("-" * 40)
