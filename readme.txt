ONGC AI Assistant - Deployment Guide

1. Install Python 3.11

2. Install Ollama:
https://ollama.com

3. Open terminal inside project folder

4. Install dependencies:
pip install -r requirements.txt

5. Pull LLM model:
ollama pull llama3

6. To add new PDFs:
Place PDFs inside:
files/

Then run:
python setup_db.py

7. Run chatbot:
python app.py

8. Open browser:
http://127.0.0.1:5050