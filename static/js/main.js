/* ==========================================================================
   Twitter Sentiment Analysis Dashboard - Cyber-Luxe JS Engine
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    
    // --- STATE VARIABLES ---
    let predictionHistory = JSON.parse(localStorage.getItem('prediction_history') || '[]');
    let currentBatchResults = [];
    let batchCurrentPage = 1;
    const batchPageSize = 10;
    let isTraining = false;
    let trainingPollInterval = null;
    let typingTimer = null;
    const typingDebounce = 400; // ms
    
    // Chart instances
    let modelChart = null;
    let classChart = null;
    let posWordsChart = null;
    let negWordsChart = null;

    // --- DOM ELEMENT REFERENCES ---
    // Navigation
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanels = document.querySelectorAll('.tab-panel');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    const activeModelBadge = document.getElementById('active-model-badge');
    
    // Theme Toggle
    const themeToggle = document.getElementById('theme-toggle');
    const themeText = document.getElementById('theme-text');
    
    // Sandbox Predictor
    const tweetInput = document.getElementById('tweet-input');
    const charCount = document.getElementById('char-count');
    const realtimeToggle = document.getElementById('realtime-toggle');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultDataPanel = document.getElementById('result-data-panel');
    const sentimentBadgeType = document.getElementById('sentiment-badge-type');
    const sentimentEmoji = document.getElementById('sentiment-emoji');
    const sentimentLabel = document.getElementById('sentiment-label');
    const sentimentConfidenceVal = document.getElementById('sentiment-confidence-val');
    const gaugeFillArc = document.getElementById('gauge-fill-arc');
    const gaugePercentage = document.getElementById('gauge-percentage');
    const probPosText = document.getElementById('prob-pos-text');
    const probPosBar = document.getElementById('prob-pos-bar');
    const probNegText = document.getElementById('prob-neg-text');
    const probNegBar = document.getElementById('prob-neg-bar');

    // Keyword Highlighting Elements
    const highlightedTweetText = document.getElementById('highlighted-tweet-text');
    const keywordTagsContainer = document.getElementById('keyword-tags-container');
    
    const historyTableBody = document.getElementById('history-table-body');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const downloadHistoryBtn = document.getElementById('download-history-btn');

    // Sentiment Battle (Compare) Elements
    const compareInputA = document.getElementById('compare-input-a');
    const compareInputB = document.getElementById('compare-input-b');
    const runCompareBtn = document.getElementById('run-compare-btn');
    const compareSummaryCard = document.getElementById('compare-summary-card');
    const compareWinnerBadge = document.getElementById('compare-winner-badge');
    
    // Battle Card A Elements
    const battleBadgeA = document.getElementById('battle-badge-a');
    const battleEmojiA = document.getElementById('battle-emoji-a');
    const battleLabelA = document.getElementById('battle-label-a');
    const battleConfA = document.getElementById('battle-conf-a');
    const battleGaugeFillA = document.getElementById('battle-gauge-fill-a');
    const battleGaugeValA = document.getElementById('battle-gauge-val-a');
    const battlePosTextA = document.getElementById('battle-pos-text-a');
    const battlePosBarA = document.getElementById('battle-pos-bar-a');
    const battleNegTextA = document.getElementById('battle-neg-text-a');
    const battleNegBarA = document.getElementById('battle-neg-bar-a');

    // Battle Card B Elements
    const battleBadgeB = document.getElementById('battle-badge-b');
    const battleEmojiB = document.getElementById('battle-emoji-b');
    const battleLabelB = document.getElementById('battle-label-b');
    const battleConfB = document.getElementById('battle-conf-b');
    const battleGaugeFillB = document.getElementById('battle-gauge-fill-b');
    const battleGaugeValB = document.getElementById('battle-gauge-val-b');
    const battlePosTextB = document.getElementById('battle-pos-text-b');
    const battlePosBarB = document.getElementById('battle-pos-bar-b');
    const battleNegTextB = document.getElementById('battle-neg-text-b');
    const battleNegBarB = document.getElementById('battle-neg-bar-b');

    // Batch CSV
    const csvDropZone = document.getElementById('csv-drop-zone');
    const csvFileInput = document.getElementById('csv-file-input');
    const fileInfoContainer = document.getElementById('file-info-container');
    const selectedFileName = document.getElementById('selected-file-name');
    const selectedFileSize = document.getElementById('selected-file-size');
    const removeFileBtn = document.getElementById('remove-file-btn');
    const processCsvBtn = document.getElementById('process-csv-btn');
    
    const batchProgressCard = document.getElementById('batch-progress-card');
    const batchProgressBar = document.getElementById('batch-progress-bar');
    
    const batchResultsSummary = document.getElementById('batch-results-summary');
    const batchPosPct = document.getElementById('batch-pos-pct');
    const batchPosCount = document.getElementById('batch-pos-count');
    const batchNegPct = document.getElementById('batch-neg-pct');
    const batchNegCount = document.getElementById('batch-neg-count');
    const batchTotalCount = document.getElementById('batch-total-count');
    const downloadBatchCsvBtn = document.getElementById('download-batch-csv-btn');
    
    const batchTableCard = document.getElementById('batch-table-card');
    const batchTableBody = document.getElementById('batch-table-body');
    const batchPrevPageBtn = document.getElementById('batch-prev-page');
    const batchNextPageBtn = document.getElementById('batch-next-page');
    const batchPageInfo = document.getElementById('batch-page-info');

    // Dashboard Info
    const dashActiveModel = document.getElementById('dash-active-model');
    const dashBestAccuracy = document.getElementById('dash-best-accuracy');
    const dashTrainingSize = document.getElementById('dash-training-size');
    const dashVocabularySize = document.getElementById('dash-vocabulary-size');
    
    // Live Images
    const imgWordcloudPos = document.getElementById('img-wordcloud-pos');
    const imgWordcloudNeg = document.getElementById('img-wordcloud-neg');
    const imgTweetLength = document.getElementById('img-tweet-length');
    const imgConfusionMatrix = document.getElementById('img-confusion-matrix');
    const staticClassDistribution = document.getElementById('static-class-distribution');

    // Training Console
    const startTrainingBtn = document.getElementById('start-training-btn');
    const trainSampleSize = document.getElementById('train-sample-size');
    const trainTuneToggle = document.getElementById('train-tune-toggle');
    const terminalLogs = document.getElementById('terminal-logs');
    const trainingStatusIndicator = document.getElementById('training-status-indicator');
    const metricsTableBody = document.getElementById('metrics-table-body');

    // --- TAB SWITCHING NAVIGATION ---
    const tabMetaData = {
        'tab-predict': { title: 'Sentiment Sandbox', desc: 'Analyze individual tweets in real-time with machine learning.' },
        'tab-compare': { title: 'Sentiment Battle', desc: 'Compare the sentiment and confidence of two tweets head-to-head.' },
        'tab-batch': { title: 'Batch CSV Mode', desc: 'Upload CSV datasets of tweets and export predictions.' },
        'tab-dashboard': { title: 'Analytics Dashboard', desc: 'Deep dive into model performance metrics and dataset word distributions.' },
        'tab-console': { title: 'Training Console', desc: 'Trigger asynchronous model retraining and monitor logs.' }
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            tabPanels.forEach(panel => {
                if (panel.id === targetTab) {
                    panel.classList.add('active');
                } else {
                    panel.classList.remove('active');
                }
            });

            if (tabMetaData[targetTab]) {
                pageTitle.textContent = tabMetaData[targetTab].title;
                pageSubtitle.textContent = tabMetaData[targetTab].desc;
            }
            
            if (targetTab === 'tab-dashboard') {
                loadDashboardStats();
            }
        });
    });

    // --- LIGHT/DARK THEME SWITCHING ---
    themeToggle.addEventListener('change', () => {
        if (themeToggle.checked) {
            document.body.classList.remove('light-theme');
            document.body.classList.add('dark-theme');
            themeText.textContent = "Dark Mode";
        } else {
            document.body.classList.remove('dark-theme');
            document.body.classList.add('light-theme');
            themeText.textContent = "Light Mode";
        }
        if (document.getElementById('tab-dashboard').classList.contains('active')) {
            loadDashboardStats();
        }
    });

    // --- CHARACTER COUNTING FOR TWEETS ---
    tweetInput.addEventListener('input', () => {
        const len = tweetInput.value.length;
        charCount.textContent = len;
        
        if (len > 250) {
            charCount.style.color = '#f43f5e';
        } else if (len > 200) {
            charCount.style.color = '#f59e0b';
        } else {
            charCount.style.color = 'var(--text-secondary)';
        }
    });

    // --- SANDBOX PREDICTION LOGIC ---
    const updateGauge = (gaugeFillEl, gaugeValEl, pct, label) => {
        const circumference = 125.6;
        const offset = circumference - (pct * circumference);
        
        gaugeFillEl.setAttribute('stroke-dashoffset', offset);
        gaugeValEl.textContent = Math.round(pct * 100) + '%';
        
        gaugeFillEl.classList.remove('positive', 'negative');
        if (label === 'Positive') {
            gaugeFillEl.classList.add('positive');
        } else if (label === 'Negative') {
            gaugeFillEl.classList.add('negative');
        }
    };

    const displayPredictionResult = (res, rawText) => {
        resultPlaceholder.classList.add('hidden');
        resultDataPanel.classList.remove('hidden');
        
        const label = res.label;
        const confidence = res.confidence;
        const probPos = res.probabilities.Positive;
        const probNeg = res.probabilities.Negative;
        
        sentimentBadgeType.className = 'card-badge';
        sentimentEmoji.className = 'emoji-container';
        sentimentLabel.className = 'sentiment-label';
        
        if (label === 'Positive') {
            sentimentBadgeType.classList.add('positive');
            sentimentBadgeType.textContent = 'Positive';
            sentimentEmoji.textContent = '😊';
            sentimentLabel.classList.add('positive');
            sentimentLabel.textContent = 'POSITIVE';
        } else if (label === 'Negative') {
            sentimentBadgeType.classList.add('negative');
            sentimentBadgeType.textContent = 'Negative';
            sentimentEmoji.textContent = '😢';
            sentimentLabel.classList.add('negative');
            sentimentLabel.textContent = 'NEGATIVE';
        } else {
            sentimentBadgeType.classList.add('awaiting');
            sentimentBadgeType.textContent = 'Unknown';
            sentimentEmoji.textContent = '😐';
            sentimentLabel.textContent = 'NEUTRAL';
        }
        
        sentimentConfidenceVal.textContent = (confidence * 100).toFixed(2) + '%';
        updateGauge(gaugeFillArc, gaugePercentage, confidence, label);
        
        // Prob bars
        probPosText.textContent = (probPos * 100).toFixed(1) + '%';
        probPosBar.style.width = (probPos * 100) + '%';
        
        probNegText.textContent = (probNeg * 100).toFixed(1) + '%';
        probNegBar.style.width = (probNeg * 100) + '%';

        // Render Keyword Highlighting
        if (res.keywords) {
            const posWords = (res.keywords.positive_words || []).map(k => k.word);
            const negWords = (res.keywords.negative_words || []).map(k => k.word);

            let htmlText = rawText.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            
            posWords.forEach(w => {
                const regex = new RegExp(`\\b(${w})\\b`, 'gi');
                htmlText = htmlText.replace(regex, `<span class="kw-highlight-pos">$1</span>`);
            });
            negWords.forEach(w => {
                const regex = new RegExp(`\\b(${w})\\b`, 'gi');
                htmlText = htmlText.replace(regex, `<span class="kw-highlight-neg">$1</span>`);
            });

            highlightedTweetText.innerHTML = htmlText;

            let tagsHtml = '';
            res.keywords.positive_words.forEach(k => {
                tagsHtml += `<span class="kw-tag pos">+ ${k.word} (${k.score})</span>`;
            });
            res.keywords.negative_words.forEach(k => {
                tagsHtml += `<span class="kw-tag neg">- ${k.word} (${k.score})</span>`;
            });
            keywordTagsContainer.innerHTML = tagsHtml || `<span class="text-secondary text-sm">No strong keyword triggers detected.</span>`;
        }
    };

    const sandboxModelSelect = document.getElementById('sandbox-model-select');
    if (sandboxModelSelect) {
        sandboxModelSelect.addEventListener('change', () => {
            if (tweetInput.value.trim()) {
                runSinglePrediction();
            }
        });
    }

    const runSinglePrediction = () => {
        const text = tweetInput.value.trim();
        if (!text) return;
        
        const modelType = sandboxModelSelect ? sandboxModelSelect.value : 'hybrid';
        
        predictBtn.disabled = true;
        predictBtn.textContent = "Analyzing...";
        resultPlaceholder.classList.remove('hidden');
        resultDataPanel.classList.add('hidden');
        resultPlaceholder.querySelector('p').textContent = "Analyzing sentiment...";

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000);

        fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, model_type: modelType }),
            signal: controller.signal
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.message); });
            }
            return response.json();
        })
        .then(data => {
            const result = data.data;
            displayPredictionResult(result, text);
            
            const modelLabel = sandboxModelSelect ? sandboxModelSelect.options[sandboxModelSelect.selectedIndex].text.split(' ')[1] : 'Hybrid';
            addToHistory(text, result.label, result.confidence, modelLabel);
        })
        .catch(err => {
            console.error("Prediction Error:", err);
            const message = err.name === 'AbortError'
                ? "Prediction timed out. Restart the Flask app and wait until it prints 'Sentiment model ready'."
                : (err.message || "Prediction failed.");
            alert("Error: " + message);
            resultPlaceholder.classList.remove('hidden');
            resultDataPanel.classList.add('hidden');
            resultPlaceholder.querySelector('p').textContent = "Write some text on the left to analyze the sentiment.";
        })
        .finally(() => {
            clearTimeout(timeoutId);
            predictBtn.disabled = false;
            predictBtn.textContent = "Predict Sentiment";
        });
    };

    tweetInput.addEventListener('keyup', () => {
        if (!realtimeToggle.checked) return;
        
        clearTimeout(typingTimer);
        const text = tweetInput.value.trim();
        
        if (text) {
            typingTimer = setTimeout(runSinglePrediction, typingDebounce);
        } else {
            resultPlaceholder.classList.remove('hidden');
            resultDataPanel.classList.add('hidden');
            sentimentBadgeType.className = 'card-badge awaiting';
            sentimentBadgeType.textContent = 'Awaiting Input';
        }
    });

    predictBtn.addEventListener('click', runSinglePrediction);

    clearBtn.addEventListener('click', () => {
        tweetInput.value = '';
        charCount.textContent = '0';
        charCount.style.color = 'var(--text-secondary)';
        resultPlaceholder.classList.remove('hidden');
        resultDataPanel.classList.add('hidden');
        resultPlaceholder.querySelector('p').textContent = "Write some text on the left to analyze the sentiment.";
        sentimentBadgeType.className = 'card-badge awaiting';
        sentimentBadgeType.textContent = 'Awaiting Input';
    });

    // --- SENTIMENT BATTLE (RESULT CARDS MATCHING USER SCREENSHOT) ---
    const populateBattleCard = (res, badgeEl, emojiEl, labelEl, confEl, gaugeFillEl, gaugeValEl, posTextEl, posBarEl, negTextEl, negBarEl) => {
        const label = res.label;
        const confidence = res.confidence;
        const probPos = res.probabilities.Positive;
        const probNeg = res.probabilities.Negative;

        badgeEl.className = 'card-badge ' + (label === 'Positive' ? 'positive' : 'negative');
        badgeEl.textContent = label;

        emojiEl.textContent = label === 'Positive' ? '😊' : '😢';

        labelEl.className = 'sentiment-label ' + (label === 'Positive' ? 'positive' : 'negative');
        labelEl.textContent = label.toUpperCase();

        confEl.textContent = (confidence * 100).toFixed(2) + '%';

        updateGauge(gaugeFillEl, gaugeValEl, confidence, label);

        posTextEl.textContent = (probPos * 100).toFixed(1) + '%';
        posBarEl.style.width = (probPos * 100) + '%';

        negTextEl.textContent = (probNeg * 100).toFixed(1) + '%';
        negBarEl.style.width = (probNeg * 100) + '%';
    };

    if (runCompareBtn) {
        runCompareBtn.addEventListener('click', () => {
            const textA = compareInputA.value.trim();
            const textB = compareInputB.value.trim();

            if (!textA || !textB) {
                alert("Please enter text for both Contender A and Contender B to launch the Sentiment Battle!");
                return;
            }

            runCompareBtn.disabled = true;
            runCompareBtn.textContent = "⚔️ Launching Battle...";

            fetch('/api/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text_a: textA, text_b: textB, model_type: 'hybrid' })
            })
            .then(res => {
                if (!res.ok) return res.json().then(e => { throw new Error(e.message); });
                return res.json();
            })
            .then(data => {
                const battleData = data.data;
                const resA = battleData.result_a;
                const resB = battleData.result_b;

                compareSummaryCard.classList.remove('hidden');

                if (battleData.winner === 'Tweet A') {
                    compareWinnerBadge.textContent = "Winner: Tweet A";
                    compareWinnerBadge.className = "card-badge positive";
                } else if (battleData.winner === 'Tweet B') {
                    compareWinnerBadge.textContent = "Winner: Tweet B";
                    compareWinnerBadge.className = "card-badge positive";
                } else {
                    compareWinnerBadge.textContent = "Outcome: Tie Match";
                    compareWinnerBadge.className = "card-badge info";
                }

                // Populate Tweet A Result Card
                populateBattleCard(
                    resA,
                    battleBadgeA, battleEmojiA, battleLabelA, battleConfA,
                    battleGaugeFillA, battleGaugeValA,
                    battlePosTextA, battlePosBarA, battleNegTextA, battleNegBarA
                );

                // Populate Tweet B Result Card
                populateBattleCard(
                    resB,
                    battleBadgeB, battleEmojiB, battleLabelB, battleConfB,
                    battleGaugeFillB, battleGaugeValB,
                    battlePosTextB, battlePosBarB, battleNegTextB, battleNegBarB
                );

                compareSummaryCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            })
            .catch(err => {
                alert("Sentiment Battle Failed: " + err.message);
            })
            .finally(() => {
                runCompareBtn.disabled = false;
                runCompareBtn.textContent = "⚔️ Launch Sentiment Battle";
            });
        });
    }

    // --- PREDICTION HISTORY ENGINE ---
    const addToHistory = (text, label, confidence, model = 'Hybrid') => {
        const timestamp = new Date().toLocaleTimeString() + ' ' + new Date().toLocaleDateString();
        
        predictionHistory.unshift({
            text: text,
            model: model,
            label: label,
            confidence: confidence,
            timestamp: timestamp
        });
        
        if (predictionHistory.length > 30) {
            predictionHistory.pop();
        }
        
        localStorage.setItem('prediction_history', JSON.stringify(predictionHistory));
        renderHistoryTable();
    };

    const renderHistoryTable = () => {
        if (predictionHistory.length === 0) {
            historyTableBody.innerHTML = `<tr><td colspan="5" class="text-center text-secondary">No predictions run yet.</td></tr>`;
            return;
        }
        
        historyTableBody.innerHTML = predictionHistory.map(item => {
            const safeText = item.text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            
            return `
                <tr>
                    <td title="${safeText}">${safeText.substring(0, 70)}${safeText.length > 70 ? '...' : ''}</td>
                    <td class="text-secondary text-sm">${item.model || 'Hybrid'}</td>
                    <td><span class="${item.label === 'Positive' ? 'card-badge positive' : 'card-badge negative'}">${item.label}</span></td>
                    <td><strong>${(item.confidence * 100).toFixed(2)}%</strong></td>
                    <td class="text-secondary text-sm">${item.timestamp}</td>
                </tr>
            `;
        }).join('');
    };

    clearHistoryBtn.addEventListener('click', () => {
        if (confirm("Are you sure you want to clear your local prediction history?")) {
            predictionHistory = [];
            localStorage.setItem('prediction_history', JSON.stringify(predictionHistory));
            renderHistoryTable();
        }
    });

    const downloadCSV = (headers, rows, filename) => {
        let csvLines = [headers.join(",")];
        rows.forEach(row => {
            let escapedRow = row.map(v => {
                let val = String(v).replace(/"/g, '""');
                return val.includes(",") || val.includes("\n") || val.includes('"') ? `"${val}"` : val;
            });
            csvLines.push(escapedRow.join(","));
        });
        const csvContent = csvLines.join("\n");
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    downloadHistoryBtn.addEventListener('click', () => {
        if (predictionHistory.length === 0) {
            alert("History is empty.");
            return;
        }
        const headers = ["Original Text", "Model Selected", "Sentiment Prediction", "Confidence Score", "Timestamp"];
        const rows = predictionHistory.map(item => [item.text, item.model || 'Hybrid', item.label, item.confidence, item.timestamp]);
        downloadCSV(headers, rows, "sentiment_history_report.csv");
    });

    renderHistoryTable();

    // --- BATCH CSV FILE DROPAREA LOGIC ---
    const preventDefaults = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        csvDropZone.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        csvDropZone.addEventListener('dragenter', () => csvDropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        csvDropZone.addEventListener('dragleave', () => csvDropZone.classList.remove('dragover'), false);
    });

    csvDropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleCsvFile(files[0]);
        }
    });

    csvDropZone.addEventListener('click', () => {
        csvFileInput.click();
    });

    csvFileInput.addEventListener('change', () => {
        if (csvFileInput.files.length) {
            handleCsvFile(csvFileInput.files[0]);
        }
    });

    const handleCsvFile = (file) => {
        if (!file.name.endsWith('.csv')) {
            alert("Please upload a valid .csv file.");
            return;
        }
        
        selectedFileName.textContent = file.name;
        selectedFileSize.textContent = (file.size / 1024).toFixed(1) + ' KB';
        
        fileInfoContainer.classList.remove('hidden');
        processCsvBtn.disabled = false;
    };

    removeFileBtn.addEventListener('click', () => {
        csvFileInput.value = '';
        fileInfoContainer.classList.add('hidden');
        processCsvBtn.disabled = true;
        
        batchResultsSummary.classList.add('hidden');
        batchTableCard.classList.add('hidden');
    });

    processCsvBtn.addEventListener('click', () => {
        const file = csvFileInput.files[0];
        if (!file) return;
        
        const batchModelSelect = document.getElementById('batch-model-select');
        const modelType = batchModelSelect ? batchModelSelect.value : 'hybrid';
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model_type', modelType);
        
        processCsvBtn.disabled = true;
        batchProgressCard.classList.remove('hidden');
        batchProgressBar.style.width = '20%';
        
        let progressVal = 20;
        const progressTimer = setInterval(() => {
            if (progressVal < 90) {
                progressVal += Math.floor(Math.random() * 15) + 5;
                batchProgressBar.style.width = progressVal + '%';
            }
        }, 150);
        
        fetch('/api/predict_batch', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            clearInterval(progressTimer);
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.message); });
            }
            return response.json();
        })
        .then(data => {
            batchProgressBar.style.width = '100%';
            
            setTimeout(() => {
                batchProgressCard.classList.add('hidden');
                
                currentBatchResults = data.results;
                batchCurrentPage = 1;
                
                const stats = data.stats;
                batchPosPct.textContent = stats.positive_percentage + '%';
                batchPosCount.textContent = stats.positive_count + ' tweets';
                batchNegPct.textContent = stats.negative_percentage + '%';
                batchNegCount.textContent = stats.negative_count + ' tweets';
                batchTotalCount.textContent = stats.total_tweets;
                
                batchResultsSummary.classList.remove('hidden');
                batchTableCard.classList.remove('hidden');
                
                renderBatchTable();
            }, 300);
        })
        .catch(err => {
            clearInterval(progressTimer);
            batchProgressCard.classList.add('hidden');
            alert("Batch prediction failed: " + err.message);
            processCsvBtn.disabled = false;
        });
    });

    const renderBatchTable = () => {
        const totalRows = currentBatchResults.length;
        const maxPages = Math.ceil(totalRows / batchPageSize);
        
        batchPageInfo.textContent = `Page ${batchCurrentPage} of ${maxPages}`;
        
        batchPrevPageBtn.disabled = batchCurrentPage === 1;
        batchNextPageBtn.disabled = batchCurrentPage === maxPages || maxPages === 0;
        
        const startIdx = (batchCurrentPage - 1) * batchPageSize;
        const pageData = currentBatchResults.slice(startIdx, startIdx + batchPageSize);
        
        batchTableBody.innerHTML = pageData.map((item, idx) => {
            const absoluteRowIndex = startIdx + idx + 1;
            const badgeClass = item.label === 'Positive' ? 'card-badge positive' : 'card-badge negative';
            
            return `
                <tr>
                    <td>${absoluteRowIndex}</td>
                    <td class="text-sm" title="${item.text}">${item.text.substring(0, 90)}${item.text.length > 90 ? '...' : ''}</td>
                    <td><span class="${badgeClass}">${item.label}</span></td>
                    <td><strong>${(item.confidence * 100).toFixed(2)}%</strong></td>
                </tr>
            `;
        }).join('');
    };

    batchPrevPageBtn.addEventListener('click', () => {
        if (batchCurrentPage > 1) {
            batchCurrentPage--;
            renderBatchTable();
        }
    });

    batchNextPageBtn.addEventListener('click', () => {
        const maxPages = Math.ceil(currentBatchResults.length / batchPageSize);
        if (batchCurrentPage < maxPages) {
            batchCurrentPage++;
            renderBatchTable();
        }
    });

    downloadBatchCsvBtn.addEventListener('click', () => {
        if (currentBatchResults.length === 0) return;
        const headers = ["Original Tweet", "Sentiment Prediction", "Confidence Score"];
        const rows = currentBatchResults.map(item => [item.text, item.label, item.confidence]);
        downloadCSV(headers, rows, "batch_sentiment_predictions.csv");
    });

    // --- PERFORMANCE ANALYTICS DASHBOARD ENGINE ---
    const loadDashboardStats = () => {
        fetch('/api/stats')
        .then(res => res.json())
        .then(data => {
            const modelMeta = data.model_comparison;
            const wordFreqs = data.word_frequencies;
            
            const d = new Date().getTime();
            imgWordcloudPos.src = `/screenshots/wordcloud_positive.png?t=${d}`;
            imgWordcloudNeg.src = `/screenshots/wordcloud_negative.png?t=${d}`;
            imgTweetLength.src = `/screenshots/tweet_length_distribution.png?t=${d}`;
            imgConfusionMatrix.src = `/screenshots/confusion_matrix.png?t=${d}`;
            staticClassDistribution.src = `/screenshots/class_distribution.png?t=${d}`;
            
            if (modelMeta) {
                activeModelBadge.textContent = "Best Model: " + modelMeta.best_model;
                dashActiveModel.textContent = modelMeta.best_model;
                dashVocabularySize.textContent = "Vocabulary size: " + modelMeta.vocabulary_size + " features";
                dashTrainingSize.textContent = modelMeta.sample_size.toLocaleString();
                
                let maxAcc = 0;
                Object.values(modelMeta.results).forEach(m => {
                    if (m.accuracy > maxAcc) maxAcc = m.accuracy;
                });
                dashBestAccuracy.textContent = (maxAcc * 100).toFixed(2) + '%';
                
                buildDetailedReportTable(modelMeta.results, modelMeta.best_model);
                buildChartJSModelComparison(modelMeta.results);
                buildChartJSClassDistribution(modelMeta.results[modelMeta.best_model].confusion_matrix);
            }
            
            if (wordFreqs) {
                buildChartJSWordFrequencies(wordFreqs.positive, 'positive-words-chart', '#06b6d4');
                buildChartJSWordFrequencies(wordFreqs.negative, 'negative-words-chart', '#f43f5e');
            }
        })
        .catch(err => {
            console.error("Dashboard stats failed to load:", err);
        });
    };

    const buildDetailedReportTable = (results, bestModelName) => {
        metricsTableBody.innerHTML = Object.entries(results).map(([name, m]) => {
            const isBest = name === bestModelName;
            const valF1Str = m.val_f1_score ? (m.val_f1_score * 100).toFixed(2) + '%' : 'N/A';
            return `
                <tr>
                    <td><strong>${name}</strong></td>
                    <td>${valF1Str}</td>
                    <td>${(m.accuracy * 100).toFixed(2)}%</td>
                    <td>${(m.precision * 100).toFixed(2)}%</td>
                    <td>${(m.recall * 100).toFixed(2)}%</td>
                    <td>${(m.f1_score * 100).toFixed(2)}%</td>
                    <td>${isBest ? '<span class="badge-selected">Selected Best</span>' : '<span class="text-secondary text-sm">Backup</span>'}</td>
                </tr>
            `;
        }).join('');
    };

    const isDark = () => {
        return !document.body.classList.contains('light-theme');
    };

    const getThemeColors = () => {
        return {
            gridLines: isDark() ? 'rgba(255, 255, 255, 0.05)' : 'rgba(15, 23, 42, 0.05)',
            text: isDark() ? '#94a3b8' : '#64748b',
            titleText: isDark() ? '#f8fafc' : '#0f172a'
        };
    };

    const buildChartJSModelComparison = (results) => {
        const ctx = document.getElementById('model-comparison-chart').getContext('2d');
        if (modelChart) modelChart.destroy();
        
        const models = Object.keys(results);
        const accuracies = models.map(m => results[m].accuracy);
        const f1s = models.map(m => results[m].f1_score);
        const theme = getThemeColors();
        
        modelChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: models,
                datasets: [
                    {
                        label: 'Accuracy',
                        data: accuracies,
                        backgroundColor: 'rgba(99, 102, 241, 0.8)',
                        borderColor: '#6366f1',
                        borderWidth: 1,
                        borderRadius: 6
                    },
                    {
                        label: 'F1-Score',
                        data: f1s,
                        backgroundColor: 'rgba(6, 182, 212, 0.8)',
                        borderColor: '#06b6d4',
                        borderWidth: 1,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: theme.text }
                    },
                    y: {
                        grid: { color: theme.gridLines },
                        ticks: { color: theme.text },
                        min: 0.4,
                        max: 1.0
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: theme.text }
                    }
                }
            }
        });
    };

    const buildChartJSClassDistribution = (cm) => {
        const ctx = document.getElementById('class-distribution-chart').getContext('2d');
        if (classChart) classChart.destroy();
        
        const theme = getThemeColors();
        const negCount = cm[0][0] + cm[0][1];
        const posCount = cm[1][0] + cm[1][1];
        
        classChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Negative', 'Positive'],
                datasets: [{
                    data: [negCount, posCount],
                    backgroundColor: ['#f43f5e', '#10b981'],
                    borderColor: isDark() ? '#080c14' : '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: theme.text }
                    }
                },
                cutout: '65%'
            }
        });
    };

    const buildChartJSWordFrequencies = (wordData, canvasId, color) => {
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        if (canvasId === 'positive-words-chart' && posWordsChart) posWordsChart.destroy();
        if (canvasId === 'negative-words-chart' && negWordsChart) negWordsChart.destroy();
        
        const words = wordData.map(w => w.word);
        const counts = wordData.map(w => w.count);
        const theme = getThemeColors();
        
        const newChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: words,
                datasets: [{
                    label: 'Word Count',
                    data: counts,
                    backgroundColor: color + 'bf',
                    borderColor: color,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        grid: { color: theme.gridLines },
                        ticks: { color: theme.text }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: theme.text }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
        
        if (canvasId === 'positive-words-chart') posWordsChart = newChart;
        if (canvasId === 'negative-words-chart') negWordsChart = newChart;
    };

    // --- TRAINING LIVE RETRAIN ENGINE ---
    const pollTrainingStatus = () => {
        fetch('/api/train_status')
        .then(res => res.json())
        .then(data => {
            const running = data.running;
            const wasTraining = isTraining;
            isTraining = running;
            
            if (running) {
                trainingStatusIndicator.className = 'status-indicator training';
                trainingStatusIndicator.textContent = 'Training...';
                startTrainingBtn.disabled = true;
                startTrainingBtn.textContent = 'Training running...';
                
                fetchLogs();
            } else {
                trainingStatusIndicator.className = 'status-indicator idle';
                trainingStatusIndicator.textContent = 'Idle';
                startTrainingBtn.disabled = false;
                startTrainingBtn.textContent = 'Trigger Retraining';
                
                if (wasTraining && trainingPollInterval) {
                    clearInterval(trainingPollInterval);
                    trainingPollInterval = null;
                    fetchLogs();
                    alert("Model training pipeline complete! Reloading newly trained model.");
                    
                    loadDashboardStats();
                }
            }
        })
        .catch(err => console.error("Poll training status failed:", err));
    };

    const fetchLogs = () => {
        fetch('/api/logs')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                terminalLogs.textContent = data.logs;
                terminalLogs.scrollTop = terminalLogs.scrollHeight;
            }
        })
        .catch(err => console.error("Failed to fetch logs:", err));
    };

    startTrainingBtn.addEventListener('click', () => {
        const size = parseInt(trainSampleSize.value);
        if (isNaN(size) || size < 1000 || size > 1600000) {
            alert("Please enter a valid sample size between 1,000 and 1,600,000.");
            return;
        }
        
        const tune = trainTuneToggle.checked;
        
        if (confirm(`Start retraining the ML models on ${size.toLocaleString()} tweets with 70% Train / 10% Review / 20% Test split?`)) {
            fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sample_size: size, tune: tune })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    terminalLogs.textContent = "[System] Retraining pipeline initiated with 70/10/20 data split. Output logs:\n\n";
                    pollTrainingStatus();
                    trainingPollInterval = setInterval(pollTrainingStatus, 1500);
                } else {
                    alert("Failed to start training: " + data.message);
                }
            })
            .catch(err => {
                alert("Error triggering training: " + err.message);
            });
        }
    });

    // --- APPLICATION STARTUP INITS ---
    const initApp = () => {
        fetch('/api/stats')
        .then(res => res.json())
        .then(data => {
            if (data.model_comparison) {
                activeModelBadge.textContent = "Best Model: " + data.model_comparison.best_model;
            } else {
                activeModelBadge.textContent = "Best Model: None Trained";
            }
        })
        .catch(() => {
            activeModelBadge.textContent = "Best Model: None Trained";
        });
        
        pollTrainingStatus();
    };

    initApp();
});
