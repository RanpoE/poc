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


