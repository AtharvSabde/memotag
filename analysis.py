from google import genai
import os
import json
import csv

# Initialize Gemini client
client = genai.Client(api_key="AIzaSyAIR--poC-8O8WBNY1AiTVl2pAz7FNulmM")

# Load the transcriptions file
with open(r"C:\Users\Admin\Desktop\memotag_test\transcibe\all_transcriptions.txt", encoding="utf-8") as f:
    transcriptions_text = f.read()

# Create prompt with the transcriptions
prompt = f"""
Here are audio transcriptions:

{transcriptions_text}

Please analyze these transcriptions and extract the following features from each sample. For EVERY feature, provide a numeric value - NO MISSING VALUES ALLOWED. If a feature cannot be measured from the given text, use a reasonable default value (0 for counts, averages for measurements, False for boolean values):

1. Fluency and Hesitation Features
- pause_count: Count the number of silent pauses (integer, default 0)
- avg_pause_duration: Average duration of pauses in seconds (float, default 0.0)
- hesitation_markers: Count of filler words like "uh", "um", "like" (integer, default 0)
- hesitation_ratio: Ratio of hesitation markers to total words (float between 0-1, default 0.0)
- word_recall_issues: Instances of self-corrections or word finding difficulties (integer, default 0)

2. Prosodic and Temporal Features
- speech_rate: Words per minute - calculate by counting words and estimating time (integer, default 120)
- pitch_variability: Numeric value representing pitch variation - if unavailable use 1.0 (float, default 1.0)
- monotonic_speech: Boolean value (true/false) indicating monotone speech (boolean, default False)

3. Lexical Retrieval Features
- naming_accuracy: Score from 0-1 indicating naming ability (float, default 0.5)
- lost_words_count: Number of instances where speaker struggles to find words (integer, default 0)
- word_association_score: Score from 0-100 on word association (integer, default 70)

4. Sentence Structure & Completion
- sentence_completion_accuracy: Score from 0-1 for completeness (float, default 0.8)
- completion_time: Average time to complete sentences in seconds (float, default 2.0)
- avg_sentence_length: Average number of words per sentence (float, default 15.0)
- syntactic_complexity: Rate from 1-3 (1=simple, 2=moderate, 3=complex) (integer, default 2)

Output Format:
Return a JSON array where each object represents one speech sample with the sample_id from the original data and ALL 15 features listed above. Ensure EVERY field has a valid numeric value - no nulls, no missing fields, no text values where numbers are expected.

Example output format:
[
  {{
    "sample_id": "processed_sample1",
    "pause_count": 5,
    "avg_pause_duration": 0.35,
    "hesitation_markers": 8,
    "hesitation_ratio": 0.12,
    "word_recall_issues": 2,
    "speech_rate": 110,
    "pitch_variability": 1.2,
    "monotonic_speech": false,
    "naming_accuracy": 0.85,
    "lost_words_count": 1,
    "word_association_score": 75,
    "sentence_completion_accuracy": 0.9,
    "completion_time": 2.5,
    "avg_sentence_length": 12.3,
    "syntactic_complexity": 2
  }}
]
"""

# Send to Gemini model
response = client.models.generate_content(
    model="gemini-2.0-flash", 
    contents=prompt
)

print(response.text)

# Get text content
output_text = response.text

# Save raw response as .txt backup
with open(r"C:\Users\Admin\Desktop\memotag_test\transcribe_1\raw_output.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

# Clean and trim the text to extract only the JSON array
start_index = output_text.find("[")
end_index = output_text.rfind("]") + 1
json_string = output_text[start_index:end_index].strip()

# Parse JSON and save as .json and .csv
try:
    extracted_data = json.loads(json_string)

    # Save JSON
    with open(r"C:\Users\Admin\Desktop\memotag_test\transcribe_1\extracted_features.json", "w", encoding="utf-8") as json_file:
        json.dump(extracted_data, json_file, indent=4)

    # Save CSV
    csv_path = r"C:\Users\Admin\Desktop\memotag_test\transcribe_1\extracted_features.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=extracted_data[0].keys())
        writer.writeheader()
        writer.writerows(extracted_data)

    print("✅ Saved JSON and CSV files successfully.")

except Exception as e:
    print("❌ Failed to parse Gemini response as JSON.")
    print(f"Error: {e}")