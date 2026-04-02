# How this works
1. Clone the directory and then open two terminals
2. ```cd backend``` in first terminal, then run following commands in the first terminal.
  - ```python -m venv .venv```
  - ```pip install -r requirements.txt```
  - ```uvicorn main:app --reload```
3. ```cd frontend``` in the second terminal, then run following commands in the second terminal.
  - ```python -m venv .venv```
  - ```pip install -r requirment.txt```
  - ```streamlit run streamlit_app.py```

## You can simply start the project by running backend.ps1 script then frontend.ps1
