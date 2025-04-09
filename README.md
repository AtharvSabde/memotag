Methodology Overview
The pipeline consists of four main components:

Audio Preprocessing (preprocess.py):

Background noise removal using spectral gating and high-pass filtering
Leading/trailing silence removal
Pause detection with configurable duration thresholds


Speech-to-Text Conversion (aud_to_text.py):

Audio transcription using OpenAI's Whisper model (via Hugging Face transformers)
Output in structured formats (text and CSV)


Feature Extraction (analysis.py):

Gemini API integration for sophisticated linguistic analysis
Extraction of 15 standardized features across four categories:

Fluency and hesitation markers
Prosodic and temporal characteristics
Lexical retrieval abilities
Sentence structure and completion metrics




Machine Learning Analysis (ml.py):

Primary Method: Isolation Forest for anomaly detection

Unsupervised learning approach suitable for detecting deviations from normal speech patterns
Contamination parameter set to 0.3 to identify the most anomalous 30% of samples


Feature importance analysis using Spearman correlation
Dimensionality reduction using PCA for visualization
