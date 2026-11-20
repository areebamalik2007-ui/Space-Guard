# SpaceGuard

Real-time space weather monitoring dashboard tracking solar flares (FLR), coronal mass ejections (CME), and geomagnetic storms (GST) using the NASA DONKI API.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set your NASA API key (defaults to `DEMO_KEY`):
   ```bash
   # Windows PowerShell:
   $env:NASA_API_KEY="your_api_key_here"
   ```

4. Run the dashboard:
   ```bash
   streamlit run app.py
   ```
