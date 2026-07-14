@echo off
cd /d %~dp0
pip install -r requirements.txt
python setup_db.py
pause