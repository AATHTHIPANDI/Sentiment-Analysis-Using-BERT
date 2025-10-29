# Sentiment Analysis Web App

A web interface for sentiment analysis using BERT, providing real-time sentiment scoring with an interactive gauge visualization.

## Features

- Web-based interface with real-time analysis
- Interactive sentiment gauge
- Probability distribution visualization
- Support for both CPU and GPU inference
- Color-coded results:
  - 🔴 Red for negative sentiment (scores 1-2)
  - 🟠 Orange for neutral sentiment (score 3)
  - 🟢 Green for positive sentiment (scores 4-5)

## Setup

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Web App

1. Start the Flask server:

```powershell
python app.py
```

2. Open your browser and navigate to: http://localhost:5000

## Usage

1. Enter or paste your text in the input box
2. Click "Analyze" or press Enter
3. View the results:
   - Sentiment gauge shows overall score
   - Emoji indicates sentiment category
   - Probability bars show confidence for each score

## Notes

- First run will download the model weights (requires internet)
- GPU acceleration is used automatically if available
- For production deployment, consider using a proper WSGI server like gunicorn