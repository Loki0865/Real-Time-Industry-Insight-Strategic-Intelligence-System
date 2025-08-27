# Week 1 — Environment Setup & Data Collection (VS Code)

This starter pack gives you a **minimal, working** data collection pipeline using **Google News RSS**.
You can run everything **inside VS Code**.

## 0) Requirements
- Python 3.10+
- VS Code with the **Python** extension

## 1) Open in VS Code
- Open VS Code → File → Open Folder → select this folder (`ai-intel-week1`)

## 2) Create & Activate Virtual Environment
> Windows (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> Windows (CMD):
```
python -m venv .venv
.\.venv\Scripts\activate.bat
```
> macOS/Linux (bash/zsh):
```
python -m venv .venv
source .venv/bin/activate
```

## 3) Install Dependencies
```
pip install -r requirements.txt
```

## 4) (Optional) Set up Twitter token for later
- Duplicate `.env.example` → rename to `.env`
- Put your token inside:
```
TWITTER_BEARER_TOKEN=YOUR_TOKEN_HERE
```

## 5) Run the News Collector (Week 1 deliverable ✅)
```
python src/collect_news.py
```
- Output CSV will be saved to: `data/raw/news/news_YYYYMMDD_HHMMSS.csv`

## 6) Verify
- Open the CSV in VS Code (or Excel) and check that headlines, links, and timestamps are present.

## 7) Next (for later weeks)
- `src/twitter_collect.py` is included as an **optional** script to pull tweets when you have a token.
- You can also create `notebooks/week1_data_collection.ipynb` in VS Code and copy code from the scripts for step-by-step exploration.

---

### Folder Structure
```
ai-intel-week1/
├─ data/
│  └─ raw/
│     └─ news/
├─ src/
│  ├─ collect_news.py
│  ├─ twitter_collect.py         (optional)
│  └─ config.py
├─ .env.example
├─ requirements.txt
└─ README.md
```

**Tip:** Commit this to Git (optional) so your project is versioned.
