import os

# Limit native thread pools before NumPy/sklearn load. On Windows + Python 3.14,
# importing sklearn from a Flask worker thread can deadlock and never return a prediction.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import io
import csv
import json
import threading
import subprocess
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder='templates', static_folder='static')

_predictor = None
predictor_lock = threading.Lock()

def get_predictor():
    global _predictor
    with predictor_lock:
        if _predictor is None:
            from src.predict import SentimentPredictor
            _predictor = SentimentPredictor()
        return _predictor

training_thread = None
training_lock = threading.Lock()

def run_training_async(sample_size=-1, tune=True):
    try:
        cmd = ["python", "src/train.py", "--sample-size", str(sample_size)]
        if tune:
            cmd.append("--tune")
        subprocess.run(cmd, check=True)
        
        subprocess.run(["python", "src/evaluate.py"], check=True)
        
        global _predictor
        with predictor_lock:
            _predictor = None
    except Exception as e:
        print(f"Error during async training thread: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        text = data.get("text", "").strip()
        model_type = data.get("model_type", "hybrid")
        if not text:
            return jsonify({"status": "error", "message": "Text input is empty"}), 400
            
        pred_service = get_predictor()
        res = pred_service.predict(text, model_type=model_type)
        return jsonify({"status": "success", "data": res})
    except FileNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": "Model not trained yet. Please run training using the Training Console tab."
        }), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    try:
        data = request.json or {}
        text_a = data.get("text_a", "").strip()
        text_b = data.get("text_b", "").strip()
        model_type = data.get("model_type", "hybrid")
        
        if not text_a or not text_b:
            return jsonify({"status": "error", "message": "Both Text A and Text B inputs must be provided."}), 400
            
        pred_service = get_predictor()
        comparison_result = pred_service.compare_tweets(text_a, text_b, model_type=model_type)
        
        return jsonify({
            "status": "success",
            "data": comparison_result
        })
    except FileNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": "Model not trained yet. Please run training using the Training Console tab."
        }), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
            
        file = request.files['file']
        model_type = request.form.get("model_type", "hybrid")
        
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"status": "error", "message": "Uploaded file must be a CSV"}), 400
            
        stream = io.StringIO(file.stream.read().decode("latin-1"), newline=None)
        csv_reader = csv.reader(stream)
        
        rows = list(csv_reader)
        if not rows:
            return jsonify({"status": "error", "message": "Uploaded CSV is empty"}), 400
            
        header = rows[0]
        text_col_idx = 0
        
        header_lower = [h.strip().lower() for h in header]
        for col_name in ['text', 'tweet', 'content', 'body', 'message', 'text_clean', 'clean_text']:
            if col_name in header_lower:
                text_col_idx = header_lower.index(col_name)
                break
                
        has_header = False
        if any(h in header_lower for h in ['text', 'tweet', 'id', 'polarity', 'sentiment', 'user', 'date']):
            has_header = True
            data_rows = rows[1:]
        else:
            data_rows = rows
            
        tweets = []
        for r in data_rows:
            if len(r) > text_col_idx:
                tweets.append(r[text_col_idx])
            else:
                tweets.append("")
                
        if not tweets:
            return jsonify({"status": "error", "message": "No text data found in CSV"}), 400
            
        pred_service = get_predictor()
        results = pred_service.predict_batch(tweets, model_type=model_type)
        
        total = len(results)
        pos_count = sum(1 for r in results if r['label'] == 'Positive')
        neg_count = total - pos_count
        
        stats = {
            "total_tweets": total,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_percentage": round((pos_count / total) * 100, 2) if total > 0 else 0,
            "negative_percentage": round((neg_count / total) * 100, 2) if total > 0 else 0
        }
        
        return jsonify({
            "status": "success",
            "stats": stats,
            "results": results
        })
        
    except FileNotFoundError as e:
        return jsonify({
            "status": "error", 
            "message": "Model not trained yet. Please run training using the Training Console tab."
        }), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/stats')
def stats():
    comparison_path = "models/model_comparison.json"
    word_freqs_path = "screenshots/word_frequencies.json"
    
    data = {
        "model_comparison": None,
        "word_frequencies": None
    }
    
    if os.path.exists(comparison_path):
        try:
            with open(comparison_path, 'r', encoding='utf-8') as f:
                data["model_comparison"] = json.load(f)
        except Exception as e:
            print(f"Error reading model comparison: {e}")
            
    if os.path.exists(word_freqs_path):
        try:
            with open(word_freqs_path, 'r', encoding='utf-8') as f:
                data["word_frequencies"] = json.load(f)
        except Exception as e:
            print(f"Error reading word frequencies: {e}")
            
    return jsonify(data)

@app.route('/api/train', methods=['POST'])
def trigger_train():
    global training_thread
    with training_lock:
        if training_thread and training_thread.is_alive():
            return jsonify({"status": "error", "message": "Training is already in progress."}), 400
            
        data = request.json or {}
        sample_size = data.get("sample_size", -1)
        tune = data.get("tune", False)
        
        training_thread = threading.Thread(target=run_training_async, args=(sample_size, tune))
        training_thread.start()
        
        return jsonify({"status": "success", "message": "Training pipeline initiated."})

@app.route('/api/train_status')
def train_status():
    global training_thread
    with training_lock:
        is_running = training_thread is not None and training_thread.is_alive()
        return jsonify({"running": is_running})

@app.route('/api/logs')
def get_logs():
    log_path = "models/training.log"
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return jsonify({"status": "success", "logs": "".join(lines[-150:])})
        except Exception as e:
            return jsonify({"status": "error", "logs": f"Error reading log file: {e}"})
    return jsonify({"status": "success", "logs": "No logs generated yet."})

@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    return send_from_directory('screenshots', filename)

def warmup_predictor():
    print("Loading sentiment model (first start can take ~20-40 seconds)...")
    try:
        pred_service = get_predictor()
        pred_service.predict("warmup", model_type="vader")
        print("Sentiment model ready.")
    except FileNotFoundError as e:
        print(f"Model files missing: {e}")
    except Exception as e:
        print(f"Failed to load sentiment model: {e}")


if __name__ == '__main__':
    warmup_predictor()
    # Reloader spawns a second process and can deadlock sklearn imports on Windows.
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)
