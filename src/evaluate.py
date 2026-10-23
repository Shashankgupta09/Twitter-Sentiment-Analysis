import os
import json
import collections
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from sklearn.metrics import ConfusionMatrixDisplay

from utils import get_logger, ensure_dir
from preprocess import clean_tweet

def generate_visualizations(data_path, comparison_json_path, output_dir="screenshots"):
    ensure_dir(output_dir)
    logger = get_logger("evaluation", "models/training.log")
    logger.info("Generating visualizations...")
    
    # 1. Load data for dataset analysis (we sample to speed up visualization)
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}, skipping dataset charts.")
        return
        
    df = pd.read_csv(data_path, encoding='latin-1', header=None)
    df = df[[0, 5]]
    df.columns = ['polarity', 'text']
    df = df[df.polarity != 2]
    df['polarity'] = df['polarity'].map({0: 'Negative', 4: 'Positive'})
    
    # Sample 50,000 tweets for visualization to keep it fast
    df_sample = df.sample(min(50000, len(df)), random_state=42).reset_index(drop=True)
    
    # A. Class Distribution Plot
    logger.info("Plotting class distribution...")
    class_counts = df['polarity'].value_counts()
    plt.figure(figsize=(6, 5))
    colors = ['#ff4d4d', '#4dff4d'] if class_counts.index[0] == 'Negative' else ['#4dff4d', '#ff4d4d']
    # Sleek dark aesthetic charts
    plt.style.use('dark_background')
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, colors=['#a855f7', '#6366f1'])
    plt.title("Dataset Sentiment Distribution", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150, facecolor='#0f172a')
    plt.close()
    
    # B. Tweet Length Distribution Plot
    logger.info("Plotting tweet length distribution...")
    df_sample['char_length'] = df_sample['text'].str.len()
    plt.figure(figsize=(8, 5))
    plt.hist(df_sample[df_sample.polarity == 'Positive']['char_length'], bins=50, alpha=0.6, label='Positive', color='#6366f1')
    plt.hist(df_sample[df_sample.polarity == 'Negative']['char_length'], bins=50, alpha=0.6, label='Negative', color='#f43f5e')
    plt.title("Tweet Length Distribution by Sentiment", fontsize=14)
    plt.xlabel("Number of Characters")
    plt.ylabel("Tweet Count")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tweet_length_distribution.png"), dpi=150, facecolor='#0f172a')
    plt.close()
    
    # Preprocess text for word analysis
    logger.info("Cleaning sample text for word clouds and charts...")
    df_sample['clean_text'] = df_sample['text'].apply(clean_tweet)
    
    pos_texts = " ".join(df_sample[df_sample.polarity == 'Positive']['clean_text'])
    neg_texts = " ".join(df_sample[df_sample.polarity == 'Negative']['clean_text'])
    
    # C. Word Clouds
    logger.info("Generating Word Clouds...")
    wc_params = {
        'width': 800,
        'height': 400,
        'background_color': '#0f172a',
        'max_words': 150
    }
    
    if pos_texts.strip():
        wc_pos = WordCloud(**wc_params, colormap='spring').generate(pos_texts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_pos, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud for Positive Tweets", fontsize=16, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wordcloud_positive.png"), dpi=150, facecolor='#0f172a')
        plt.close()
        
    if neg_texts.strip():
        wc_neg = WordCloud(**wc_params, colormap='winter').generate(neg_texts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_neg, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud for Negative Tweets", fontsize=16, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wordcloud_negative.png"), dpi=150, facecolor='#0f172a')
        plt.close()
        
    # D. Top Positive & Negative Words
    logger.info("Finding top positive and negative words...")
    pos_words = pos_texts.split()
    neg_words = neg_texts.split()
    
    pos_counter = collections.Counter(pos_words).most_common(15)
    neg_counter = collections.Counter(neg_words).most_common(15)
    
    # Save word counts to json for frontend loading
    word_freqs = {
        'positive': [{'word': w, 'count': c} for w, c in pos_counter],
        'negative': [{'word': w, 'count': c} for w, c in neg_counter]
    }
    with open(os.path.join(output_dir, "word_frequencies.json"), "w", encoding="utf-8") as f:
        json.dump(word_freqs, f, indent=4)
        
    # Plot top positive words
    if pos_counter:
        words, counts = zip(*pos_counter)
        plt.figure(figsize=(8, 5))
        plt.barh(words[::-1], counts[::-1], color='#3b82f6')
        plt.title("Top 15 Most Common Words in Positive Tweets", fontsize=14)
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_positive_words.png"), dpi=150, facecolor='#0f172a')
        plt.close()
        
    # Plot top negative words
    if neg_counter:
        words, counts = zip(*neg_counter)
        plt.figure(figsize=(8, 5))
        plt.barh(words[::-1], counts[::-1], color='#ef4444')
        plt.title("Top 15 Most Common Words in Negative Tweets", fontsize=14)
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_negative_words.png"), dpi=150, facecolor='#0f172a')
        plt.close()
        
    # 2. Model Comparisons (if JSON is present)
    if os.path.exists(comparison_json_path):
        logger.info("Plotting model comparison chart...")
        with open(comparison_json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        results = meta['results']
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        f1_scores = [results[m]['f1_score'] for m in models]
        
        # E. Model accuracy comparison bar chart
        x = np.arange(len(models))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, accuracies, width, label='Accuracy', color='#a855f7')
        plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='#06b6d4')
        
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Machine Learning Model Comparison', fontsize=14, pad=15)
        plt.xticks(x, models, rotation=15, ha='right')
        plt.ylim(0.4, 1.0)
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150, facecolor='#0f172a')
        plt.close()
        
        # F. Confusion Matrix of Best Model
        best_model_name = meta['best_model']
        logger.info(f"Plotting Confusion Matrix for best model: {best_model_name}")
        cm = np.array(results[best_model_name]['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(ax=ax, cmap='Purples', values_format='d')
        plt.title(f"Confusion Matrix ({best_model_name})", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, facecolor='#0f172a')
        plt.close()
        
    logger.info("All visualizations generated successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate visualizations for Twitter Sentiment Analysis.")
    parser.add_argument("--data-path", type=str, default="data/training.1600000.processed.noemoticon.csv", help="Path to csv data")
    parser.add_argument("--comparison-json", type=str, default="models/model_comparison.json", help="Path to model comparison json")
    parser.add_argument("--output-dir", type=str, default="screenshots", help="Directory to save charts")
    args = parser.parse_args()
    
    generate_visualizations(args.data_path, args.comparison_json, args.output_dir)
