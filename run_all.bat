@echo off
echo ============================================================
echo    AI-Powered Network Intrusion Detection System
echo    Complete Pipeline Runner
echo ============================================================
echo.

cd /d "C:\Capstone Project"

REM Activate virtual environment
call venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Virtual environment not found!
    echo Run: python -m venv venv
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

echo [1/7] Running Preprocessing (5-15 min)...
echo --------------------------------------------------------
python src/preprocessing.py
if errorlevel 1 (
    echo [FAILED] Preprocessing failed!
    pause
    exit /b 1
)
echo [1/7] DONE
echo.

echo [2/7] Running EDA...
echo --------------------------------------------------------
python src/eda.py
if errorlevel 1 (
    echo [FAILED] EDA failed!
    pause
    exit /b 1
)
echo [2/7] DONE
echo.

echo [3/7] Running Feature Engineering (3-5 min)...
echo --------------------------------------------------------
python src/feature_engineering.py
if errorlevel 1 (
    echo [FAILED] Feature Engineering failed!
    pause
    exit /b 1
)
echo [3/7] DONE
echo.

echo [4/7] Running Data Split + SMOTE (5-10 min)...
echo --------------------------------------------------------
python src/data_split.py
if errorlevel 1 (
    echo [FAILED] Data Split failed!
    pause
    exit /b 1
)
echo [4/7] DONE
echo.

echo [5/7] Running Model Training (15-30 min)...
echo --------------------------------------------------------
python src/train.py
if errorlevel 1 (
    echo [FAILED] Model Training failed!
    pause
    exit /b 1
)
echo [5/7] DONE
echo.

echo [6/7] Running Hyperparameter Tuning (15-30 min)...
echo --------------------------------------------------------
python src/hyperparameter_tuning.py
if errorlevel 1 (
    echo [FAILED] Hyperparameter Tuning failed!
    pause
    exit /b 1
)
echo [6/7] DONE
echo.

echo [7/7] Running Visualizations + SHAP...
echo --------------------------------------------------------
python src/visualize_results.py
if errorlevel 1 (
    echo [WARNING] Visualizations had issues, continuing...
)
python src/explainability.py
if errorlevel 1 (
    echo [WARNING] SHAP had issues, continuing...
)
echo [7/7] DONE
echo.

echo ============================================================
echo    ALL PHASES COMPLETE!
echo ============================================================
echo.
echo To launch the dashboard, run:
echo    streamlit run dashboard/app.py
echo.
pause