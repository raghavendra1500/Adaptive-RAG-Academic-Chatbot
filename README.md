It should include:

Project title
Overview
Features
Folder structure
Installation
How to run
Screenshots
Future work
License


git clone https://github.com/USERNAME/Adaptive-RAG-Academic-Chatbot.git

cd Adaptive-RAG-Academic-Chatbot

python -m venv .venv

.\.venv\Scripts\activate

pip install -r requirements.txt

ollama pull llama3.2:3b

streamlit run app.py