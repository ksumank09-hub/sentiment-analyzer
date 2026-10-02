# Deploy to Render (free) - get a public link

1. Create a free account at https://github.com and a new empty repository (e.g. `sentiment-analyzer`).
2. Upload everything inside this `sentiment_project` folder to the repository
   (GitHub -> "Add file" -> "Upload files", drag the files and folders, Commit).
   The files `app.py`, `requirements-deploy.txt`, `render.yaml` and `Procfile` must be in the repository root.
3. Create a free account at https://render.com (sign in with GitHub).
4. New + -> **Blueprint** -> pick your repository -> Apply.
   (Or New + -> Web Service, and enter:
   Build Command: `pip install -r requirements-deploy.txt && python -m src.train`
   Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   Instance type: Free)
5. Wait 3-6 minutes for the build. Render then shows your link, like https://sentiment-analyzer-xxxx.onrender.com

Notes
- Free services sleep after ~15 minutes without visitors; the first open afterwards takes 30-60 seconds.
- The link is public: anyone who has it can use the app.
