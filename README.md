Setup

```
python -m venv .venv
.venv/bin/activate
```

Install
```
pip install -r requirements.txt
```

Run
```
uvicorn asr_v2:app --host 0.0.0.0 --port 8001 --reload
```

SET .env file
```
ANTHROPIC_API_KEY=<key>
```

Frontend add the page to any route in NEXTJS project

Deploy to Render
----------------
1. Commit and push the `Dockerfile` and `render.yaml` files.
2. In Render, create a new **Web Service** and connect it to this repository.
3. Select **Docker** for the runtime; Render will use `render.yaml` (configured for the Free plan) to build and run the service.
4. Add the required environment variable `ANTHROPIC_API_KEY` under *Environment*, then deploy.
5. Once live, the service will expose the FastAPI app on the default Render hostname; note that Free instances may sleep when idle, so allow a short warm-up before use.
