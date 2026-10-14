# Autonomous AI Data Analyst

AI-powered data analysis application built with Python, Streamlit, Pandas, and NVIDIA Nemotron.

## Features
- CSV and Excel file uploads
- Dataset health and missing-value analysis
- Natural-language business questions
- Automated analysis, rankings, and statistics
- Business insights and chart generation
- Multi-turn conversations

## Requirements
Python 3.12 and an NVIDIA API key.

## Local Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
Bhai, ab ye **poora block ek saath copy-paste** karo. 👇

```bash
cd ~/projects/autonomous-ai-data-analyst
mkdir -p .streamlit

cat > README.md <<'EOF'
# Autonomous AI Data Analyst

AI-powered data analysis application built with Python, Streamlit, Pandas, and NVIDIA Nemotron.

## Features
- CSV and Excel file uploads
- Dataset health and missing-value analysis
- Natural-language business questions
- Automated analysis, rankings, and statistics
- Business insights and chart generation
- Multi-turn conversations

## Requirements
Python 3.12 and an NVIDIA API key.

## Local Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Configure the NVIDIA API credentials expected by the application. Never commit API keys.

## Deployment
Deploy on Streamlit Community Cloud using `app.py` as the entry point. Configure secrets through the platform settings.

Status: Development and deployment verification in progress.
