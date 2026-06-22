@echo off
echo ==========================================
echo   AXIS TRANSLATOR - STREAMLIT
echo ==========================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting Streamlit...
echo.
echo ==========================================
echo   Server running at: http://localhost:8501
echo ==========================================
echo.

streamlit run streamlit_app.py

pause