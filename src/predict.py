import os
import joblib
import numpy as np
try:
    from src.preprocess import clean_tweet
except ModuleNotFoundError:
    from preprocess import clean_tweet

class SentimentPredictor:
    def __init__(self, model_path="models/sentiment_model.pkl", vectorizer_path="models/vectorizer.pkl"):
        """
        Loads the saved classification model and vectorizer.
        Provides a clean, modular prediction wrapper that abstracts the underlying architecture,
        supporting ML classification models, VADER lexicon analysis, a Hybrid Ensemble,
        Sentiment Keyword Extraction, and Dual-Tweet Battle Comparison.
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.sia = None
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}. Please run train.py first.")
            
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def _get_vader_sia(self):
        if self.sia is None:
            import nltk
            try:
                nltk.data.find('sentiment/vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.sia = SentimentIntensityAnalyzer()
        return self.sia
        
    def _get_confidence(self, features):
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            return float(proba[0]), float(proba[1])
            
        elif hasattr(self.model, "decision_function"):
            decision_val = self.model.decision_function(features)[0]
            prob_positive = 1.0 / (1.0 + np.exp(-decision_val))
            prob_negative = 1.0 - prob_positive
            return float(prob_negative), float(prob_positive)
            
        else:
            return 0.5, 0.5

    def _get_confidence_matrix(self, features_matrix):
        n_samples = features_matrix.shape[0]
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(features_matrix)
            return probas
        elif hasattr(self.model, "decision_function"):
            dec_vals = self.model.decision_function(features_matrix)
            prob_pos = 1.0 / (1.0 + np.exp(-dec_vals))
            prob_neg = 1.0 - prob_pos
            return np.column_stack((prob_neg, prob_pos))
        else:
            return np.full((n_samples, 2), 0.5)

    def extract_keywords(self, text):
        sia = self._get_vader_sia()
        words = [w.strip(".,!?:;\"'()[]{}") for w in text.split()]
        pos_keywords = []
        neg_keywords = []
        
        seen = set()
        for raw_word in words:
            clean_w = raw_word.lower()
            if not clean_w or len(clean_w) <= 2 or clean_w in seen:
                continue
            seen.add(clean_w)
            
            val = sia.lexicon.get(clean_w, 0.0)
            if val > 0.5:
                pos_keywords.append({'word': raw_word, 'score': round(val, 2)})
            elif val < -0.5:
                neg_keywords.append({'word': raw_word, 'score': round(abs(val), 2)})
                
        pos_keywords.sort(key=lambda x: x['score'], reverse=True)
        neg_keywords.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'positive_words': pos_keywords[:5],
            'negative_words': neg_keywords[:5]
        }

    def predict(self, text, model_type="hybrid"):
        if not text.strip():
            return {
                'label': 'Neutral/Unknown',
                'confidence': 0.5,
                'probabilities': {'Negative': 0.5, 'Positive': 0.5},
                'keywords': {'positive_words': [], 'negative_words': []}
            }
            
        cleaned = clean_tweet(text)
        features = self.vectorizer.transform([cleaned])
        ml_neg, ml_pos = self._get_confidence(features)
        
        text_lower = text.lower()
        
        extreme_negative_words = [
            "genocide", "kill", "killing", "killed", "killer", "murder", "murdering", "murdered",
            "slaughter", "massacre", "die", "dying", "death", "dead", "deadly", "destroy",
            "destroying", "destruction", "destructive", "harm", "harming", "harmful", "hurt",
            "hurting", "abuse", "abusive", "violence", "violent", "attack", "attacking",
            "weapon", "poison", "terrorist", "terrorism", "bomb", "bombing", "crime",
            "criminal", "suicide", "evil", "disgusting", "racist", "racism", "hate",
            "hated", "hating", "annihilate", "annihilation", "extinction", "fatal",
            "lethal", "suffering", "tragedy", "catastrophe", "torture", "assassinate", "victim"
        ]
        
        negation_phrases = [
            "not good", "not great", "not happy", "not optimal", "not working", "not work",
            "no progress", "no help", "no hope", "no success", "no good", "no result", "no change",
            "not impressive", "not impressed", "thanks for nothing", "oh great",
            "great, another", "just what i needed", "waste of", "waste time", "worst",
            "terrible", "horrible", "awful", "sucks", "useless", "disappointed",
            "disappointing", "annoying", "annoyed", "ruined", "painful",
            "boring", "slow", "broken", "issue", "problem", "bug", "bad", "wrong", "fail", "failed"
        ]

        warm_positive_triggers = [
            "hug", "need a hug", "want a hug", "send hugs", "love", "grateful",
            "blessed", "awesome", "wonderful", "beautiful", "happy", "great job"
        ]
        
        has_extreme_neg = any(word in text_lower for word in extreme_negative_words)
        has_neg_trigger = any(phrase in text_lower for phrase in negation_phrases)
        has_warm_pos = any(phrase in text_lower for phrase in warm_positive_triggers)
        
        sia = self._get_vader_sia()
        vader_scores = sia.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        if has_extreme_neg:
            vader_compound = -0.90
            ml_pos = min(ml_pos, 0.10)
            ml_neg = 1.0 - ml_pos
        elif has_neg_trigger and vader_compound > -0.1:
            vader_compound = min(vader_compound - 0.5, -0.3)
            ml_pos = min(ml_pos, 0.35)
            ml_neg = 1.0 - ml_pos
        elif has_warm_pos and not has_extreme_neg and not has_neg_trigger:
            vader_compound = max(vader_compound, 0.45)
            ml_pos = max(ml_pos, 0.65)
            ml_neg = 1.0 - ml_pos
            
        vader_pos = 1.0 / (1.0 + np.exp(-3.5 * vader_compound))
        vader_neg = 1.0 - vader_pos
        
        if model_type == "vader":
            prob_neg, prob_pos = vader_neg, vader_pos
        elif model_type == "ml":
            prob_neg, prob_pos = ml_neg, ml_pos
        else:
            prob_pos = 0.45 * ml_pos + 0.55 * vader_pos
            prob_neg = 1.0 - prob_pos
            
        label = "Positive" if prob_pos >= 0.5 else "Negative"
        confidence = prob_pos if label == "Positive" else prob_neg
        
        keywords = self.extract_keywords(text)

        return {
            'label': label,
            'confidence': float(confidence),
            'probabilities': {
                'Negative': float(prob_neg),
                'Positive': float(prob_pos)
            },
            'keywords': keywords
        }

    def compare_tweets(self, text_a, text_b, model_type="hybrid"):
        """
        DUAL TWEET BATTLE ENGINE.
        Runs comparative analysis between Tweet A and Tweet B:
        - Calculates Net Positivity & Sentiment Intensity Scores
        - Extracts and contrasts positive & negative trigger words
        - Generates a smart AI-style Analytical Verdict narrative
        """
        res_a = self.predict(text_a, model_type=model_type)
        res_b = self.predict(text_b, model_type=model_type)
        
        pos_a = res_a['probabilities']['Positive']
        pos_b = res_b['probabilities']['Positive']
        
        neg_a = res_a['probabilities']['Negative']
        neg_b = res_b['probabilities']['Negative']
        
        intensity_a = round(abs(pos_a - neg_a) * 100, 1)
        intensity_b = round(abs(pos_b - neg_b) * 100, 1)
        
        diff = pos_a - pos_b
        abs_diff_pct = round(abs(diff) * 100, 1)
        
        if abs(diff) < 0.05 and res_a['label'] == res_b['label']:
            winner = 'Tie'
            winner_title = 'Equal Matchup (Tie)'
            winner_reason = f'Both tweets express virtually identical sentiment levels ({round(pos_a*100, 1)}% positive).'
        elif pos_a > pos_b:
            winner = 'Tweet A'
            winner_title = 'Tweet A Wins (More Positive)'
            winner_reason = f'Tweet A is {abs_diff_pct}% more positive than Tweet B.'
        else:
            winner = 'Tweet B'
            winner_title = 'Tweet B Wins (More Positive)'
            winner_reason = f'Tweet B is {abs_diff_pct}% more positive than Tweet A.'
            
        pos_words_a = [k['word'] for k in res_a['keywords']['positive_words']]
        neg_words_a = [k['word'] for k in res_a['keywords']['negative_words']]
        
        pos_words_b = [k['word'] for k in res_b['keywords']['positive_words']]
        neg_words_b = [k['word'] for k in res_b['keywords']['negative_words']]
        
        verdict_parts = []
        
        desc_a = f"Tweet A is classified as {res_a['label'].upper()} ({round(res_a['confidence']*100, 1)}% confidence)."
        if pos_words_a:
            desc_a += f" Key positive triggers: {', '.join(pos_words_a)}."
        if neg_words_a:
            desc_a += f" Key negative triggers: {', '.join(neg_words_a)}."
            
        desc_b = f"Tweet B is classified as {res_b['label'].upper()} ({round(res_b['confidence']*100, 1)}% confidence)."
        if pos_words_b:
            desc_b += f" Key positive triggers: {', '.join(pos_words_b)}."
        if neg_words_b:
            desc_b += f" Key negative triggers: {', '.join(neg_words_b)}."
            
        verdict_parts.append(desc_a)
        verdict_parts.append(desc_b)
        
        if winner == 'Tweet A':
            verdict_parts.append(f"VERDICT: Tweet A takes victory with a {abs_diff_pct}% higher positive sentiment margin and an overall sentiment intensity score of {intensity_a}%.")
        elif winner == 'Tweet B':
            verdict_parts.append(f"VERDICT: Tweet B takes victory with a {abs_diff_pct}% higher positive sentiment margin and an overall sentiment intensity score of {intensity_b}%.")
        else:
            verdict_parts.append("VERDICT: Both tweets show equivalent sentiment balance.")
            
        analytical_verdict = " ".join(verdict_parts)
        
        return {
            'text_a': text_a,
            'text_b': text_b,
            'result_a': res_a,
            'result_b': res_b,
            'winner': winner,
            'winner_title': winner_title,
            'winner_reason': winner_reason,
            'positivity_diff_pct': abs_diff_pct,
            'intensity_a': intensity_a,
            'intensity_b': intensity_b,
            'analytical_verdict': analytical_verdict
        }

    def predict_batch(self, texts, model_type="hybrid"):
        if not texts:
            return []
            
        cleaned_texts = [clean_tweet(t) for t in texts]
        features_matrix = self.vectorizer.transform(cleaned_texts)
        ml_probs_matrix = self._get_confidence_matrix(features_matrix)
        sia = self._get_vader_sia()
        
        results = []
        for i, text in enumerate(texts):
            if not text.strip():
                results.append({
                    'text': text,
                    'label': 'Neutral/Unknown',
                    'confidence': 0.5,
                    'probabilities': {'Negative': 0.5, 'Positive': 0.5},
                    'keywords': {'positive_words': [], 'negative_words': []}
                })
                continue
                
            ml_neg, ml_pos = ml_probs_matrix[i, 0], ml_probs_matrix[i, 1]
            
            vader_scores = sia.polarity_scores(text)
            vader_compound = vader_scores['compound']
            
            vader_pos = 1.0 / (1.0 + np.exp(-3.5 * vader_compound))
            vader_neg = 1.0 - vader_pos
            
            if model_type == "vader":
                prob_neg, prob_pos = vader_neg, vader_pos
            elif model_type == "ml":
                prob_neg, prob_pos = ml_neg, ml_pos
            else:
                prob_pos = 0.45 * ml_pos + 0.55 * vader_pos
                prob_neg = 1.0 - prob_pos
                
            label = "Positive" if prob_pos >= 0.5 else "Negative"
            confidence = prob_pos if label == "Positive" else prob_neg
            
            results.append({
                'text': text,
                'label': label,
                'confidence': float(confidence),
                'probabilities': {
                    'Negative': float(prob_neg),
                    'Positive': float(prob_pos)
                },
                'keywords': self.extract_keywords(text)
            })
            
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict tweet sentiment.")
    parser.add_argument("text", type=str, help="Text to predict sentiment for")
    parser.add_argument("--model-type", type=str, default="hybrid", choices=["hybrid", "ml", "vader"], help="Model type to use")
    args = parser.parse_args()
    
    try:
        predictor = SentimentPredictor()
        res = predictor.predict(args.text, model_type=args.model_type)
        print(f"\nPrediction Results ({args.model_type.upper()} Model):")
        print(f"Input:        '{args.text}'")
        print(f"Sentiment:    {res['label']}")
        print(f"Confidence:   {res['confidence']:.2%}")
    except Exception as e:
        print(f"Error: {e}")
