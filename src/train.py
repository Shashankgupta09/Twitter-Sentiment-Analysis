import os
import argparse
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils import get_logger, ensure_dir
from preprocess import clean_tweet

def load_and_preprocess_data(data_path, sample_size, logger):
    logger.info("Loading dataset...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")
        
    df = pd.read_csv(data_path, encoding='latin-1', header=None)
    df = df[[0, 5]]
    df.columns = ['polarity', 'text']
    
    # Keep positive (4 -> 1) and negative (0 -> 0)
    df = df[df.polarity != 2]
    df['polarity'] = df['polarity'].map({0: 0, 4: 1})
    
    counts = df['polarity'].value_counts()
    logger.info(f"Full raw dataset size: {len(df)} tweets ({counts.get(0, 0)} negative, {counts.get(1, 0)} positive)")
    
    if sample_size > 0 and sample_size < len(df):
        logger.info(f"Sampling balanced subset of {sample_size} tweets...")
        half_sample = sample_size // 2
        df_pos = df[df.polarity == 1].sample(min(half_sample, counts.get(1, 0)), random_state=42)
        df_neg = df[df.polarity == 0].sample(min(half_sample, counts.get(0, 0)), random_state=42)
        df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        logger.info("Using the full dataset (no 50k sample cap). Split will be 70% train / 10% review / 20% test.")
        
    logger.info("Cleaning tweets text...")
    df['clean_text'] = df['text'].apply(clean_tweet)
    
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    
    logger.info(f"Final cleaned dataset size: {len(df)} tweets")
    return df

def run_hyperparameter_tuning(X_train, y_train, logger, tune_sample_size=5000):
    logger.info("Starting hyperparameter tuning...")
    
    n_samples = X_train.shape[0]
    if n_samples > tune_sample_size:
        indices = np.random.choice(n_samples, tune_sample_size, replace=False)
        X_tune = X_train[indices]
        y_tune = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
    else:
        X_tune = X_train
        y_tune = y_train
        
    tuned_params = {}
    
    logger.info("Tuning Logistic Regression...")
    lr_grid = {'C': [0.1, 1.0, 10.0]}
    grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), lr_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_lr.fit(X_tune, y_tune)
    tuned_params['Logistic Regression'] = grid_lr.best_params_
    
    logger.info("Tuning Linear SVM...")
    svm_grid = {'C': [0.1, 1.0]}
    grid_svm = GridSearchCV(LinearSVC(max_iter=1000, dual='auto'), svm_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_svm.fit(X_tune, y_tune)
    tuned_params['Linear SVM'] = grid_svm.best_params_
    
    logger.info("Tuning Random Forest...")
    rf_grid = {'n_estimators': [50, 100], 'max_depth': [10, 15]}
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_rf.fit(X_tune, y_tune)
    tuned_params['Random Forest'] = grid_rf.best_params_
    
    return tuned_params

def main():
    parser = argparse.ArgumentParser(description="Train sentiment analysis models with 70/10/20 data split.")
    parser.add_argument("--data-path", type=str, default="data/training.1600000.processed.noemoticon.csv", help="Path to raw csv dataset")
    parser.add_argument("--sample-size", type=int, default=-1, help="Number of tweets to sample (-1 uses the full dataset)")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    parser.add_argument("--tune-sample-size", type=int, default=5000, help="Sample size for hyperparameter tuning")
    args = parser.parse_args()
    
    ensure_dir("models")
    ensure_dir("screenshots")
    
    logger = get_logger("training", "models/training.log")
    logger.info("Starting model training pipeline with 70% Train, 10% Review/Val, 20% Test split...")
    
    # 1. Load Data
    df = load_and_preprocess_data(args.data_path, args.sample_size, logger)
    
    # 2. Vectorization
    logger.info("Fitting TF-IDF Vectorizer (max_features=30000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )
    
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['polarity']
    
    vectorizer_path = "models/vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"Vectorizer saved to {vectorizer_path}")
    
    # 3. STRICT 70% TRAIN / 10% REVIEW / 20% TEST SPLIT
    total_samples = len(df)
    indices = np.arange(total_samples)
    
    # Step A: Split 20% for Test Set
    train_val_idx, test_idx, y_train_val, y_test = train_test_split(
        indices, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Step B: Split remaining 80% into 70% Train (0.875 of 80% = 70% of total) and 10% Review/Val (0.125 of 80% = 10% of total)
    train_idx, val_idx, y_train, y_val = train_test_split(
        train_val_idx, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
    )
    
    X_train, y_train_data = X[train_idx], y.iloc[train_idx]
    X_val, y_val_data = X[val_idx], y.iloc[val_idx]
    X_test, y_test_data = X[test_idx], y.iloc[test_idx]
    
    test_raw_texts = df.iloc[test_idx]['text'].tolist()
    
    logger.info("=" * 70)
    logger.info(f"EXACT DATASET SPLIT CONFIRMATION:")
    logger.info(f" -> 70% Training Set:         {X_train.shape[0]} samples ({X_train.shape[0]/total_samples*100:.1f}%)")
    logger.info(f" -> 10% Review/Validation Set: {X_val.shape[0]} samples ({X_val.shape[0]/total_samples*100:.1f}%)")
    logger.info(f" -> 20% Testing Set:           {X_test.shape[0]} samples ({X_test.shape[0]/total_samples*100:.1f}%)")
    logger.info("=" * 70)
    
    # 4. Hyperparameter Tuning
    if args.tune:
        tuned_params = run_hyperparameter_tuning(X_train, y_train_data, logger, args.tune_sample_size)
    else:
        tuned_params = {
            'Logistic Regression': {'C': 1.0},
            'Linear SVM': {'C': 1.0},
            'Random Forest': {'n_estimators': 100, 'max_depth': 15}
        }
        
    # 5. Define Models
    models_to_train = {
        'Bernoulli Naive Bayes': BernoulliNB(),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(C=tuned_params['Logistic Regression']['C'], max_iter=1000),
        'Linear SVM': LinearSVC(C=tuned_params['Linear SVM']['C'], max_iter=1000, dual='auto'),
        'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=tuned_params['Random Forest']['n_estimators'],
            max_depth=tuned_params['Random Forest']['max_depth'],
            random_state=42,
            n_jobs=-1
        )
    }
    
    comparison_results = {}
    best_model_name = None
    best_val_f1 = -1.0
    best_model_obj = None
    
    for name, model in models_to_train.items():
        logger.info(f"Training {name} on 70% Training set ({X_train.shape[0]} samples)...")
        model.fit(X_train, y_train_data)
        
        # Evaluate on 10% Review/Validation set
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val_data, val_preds)
        _, _, val_f1, _ = precision_recall_fscore_support(y_val_data, val_preds, average='binary')
        logger.info(f"  {name} -> 10% Review Set Acc: {val_acc:.4f}, Review F1: {val_f1:.4f}")
        
        # Evaluate on 20% Testing set
        test_preds = model.predict(X_test)
        acc = accuracy_score(y_test_data, test_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_data, test_preds, average='binary')
        cm = confusion_matrix(y_test_data, test_preds).tolist()
        
        logger.info(f"  {name} -> 20% Test Set Acc:   {acc:.4f}, Test F1:   {f1:.4f}")
        
        comparison_results[name] = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'val_accuracy': float(val_acc),
            'val_f1_score': float(val_f1),
            'confusion_matrix': cm
        }
        
        filename = name.lower().replace(" ", "_")
        model_path = f"models/model_{filename}.pkl"
        joblib.dump(model, model_path)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_name = name
            best_model_obj = model
            
    logger.info(f"Selected Best Model on 10% Review set: {best_model_name} (Val F1: {best_val_f1:.4f})")
    
    # Save best model
    best_model_path = "models/sentiment_model.pkl"
    joblib.dump(best_model_obj, best_model_path)
    logger.info(f"Saved Best Model '{best_model_name}' to {best_model_path}")
    
    # Evaluate VADER and Hybrid on test set
    try:
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        
        logger.info("Evaluating VADER model on 20% Test set...")
        vader_preds = []
        vader_probs = []
        for tweet in test_raw_texts:
            scores = sia.polarity_scores(tweet)
            compound = scores['compound']
            pos_p = 1.0 / (1.0 + np.exp(-3.5 * compound))
            neg_p = 1.0 - pos_p
            vader_probs.append([neg_p, pos_p])
            vader_preds.append(1 if pos_p >= 0.5 else 0)
            
        acc_v = accuracy_score(y_test_data, vader_preds)
        p_v, r_v, f1_v, _ = precision_recall_fscore_support(y_test_data, vader_preds, average='binary')
        cm_v = confusion_matrix(y_test_data, vader_preds).tolist()
        
        comparison_results['VADER Sentiment'] = {
            'accuracy': float(acc_v),
            'precision': float(p_v),
            'recall': float(r_v),
            'f1_score': float(f1_v),
            'val_accuracy': float(acc_v),
            'val_f1_score': float(f1_v),
            'confusion_matrix': cm_v
        }
        
        logger.info("Evaluating Hybrid Ensemble on 20% Test set...")
        if hasattr(best_model_obj, "predict_proba"):
            ml_test_probs = best_model_obj.predict_proba(X_test)
        elif hasattr(best_model_obj, "decision_function"):
            dec = best_model_obj.decision_function(X_test)
            pos_p = 1.0 / (1.0 + np.exp(-dec))
            neg_p = 1.0 - pos_p
            ml_test_probs = np.column_stack((neg_p, pos_p))
        else:
            ml_test_probs = np.ones((len(test_raw_texts), 2)) * 0.5
            
        vader_probs = np.array(vader_probs)
        hybrid_probs = 0.45 * ml_test_probs + 0.55 * vader_probs
        hybrid_preds = np.argmax(hybrid_probs, axis=1)
        
        acc_h = accuracy_score(y_test_data, hybrid_preds)
        p_h, r_h, f1_h, _ = precision_recall_fscore_support(y_test_data, hybrid_preds, average='binary')
        cm_h = confusion_matrix(y_test_data, hybrid_preds).tolist()
        
        comparison_results['Hybrid Ensemble'] = {
            'accuracy': float(acc_h),
            'precision': float(p_h),
            'recall': float(r_h),
            'f1_score': float(f1_h),
            'val_accuracy': float(acc_h),
            'val_f1_score': float(f1_h),
            'confusion_matrix': cm_h
        }
    except Exception as e:
        logger.error(f"Error evaluating VADER/Hybrid: {e}")
        
    metrics_path = "models/model_comparison.json"
    metadata = {
        'best_model': best_model_name,
        'results': comparison_results,
        'sample_size': int(len(df)),
        'vocabulary_size': X.shape[1],
        'split_ratio': {'train': 0.70, 'review': 0.10, 'test': 0.20}
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    logger.info("=" * 85)
    logger.info(f"{'Model Name':<28} | {'70% Train F1':<14} | {'10% Review F1':<14} | {'20% Test Acc':<14} | {'20% Test F1':<12}")
    logger.info("=" * 85)
    for name, metrics in comparison_results.items():
        val_f1_str = f"{metrics.get('val_f1_score', 0):.4f}"
        logger.info(f"{name:<28} | {'N/A':<14} | {val_f1_str:<14} | {metrics['accuracy']:<14.4f} | {metrics['f1_score']:<12.4f}")
    logger.info("=" * 85)

if __name__ == "__main__":
    main()
