@echo off
call venv\Scripts\activate.bat
python ml_project.py
python train_models.py
streamlit run app.py