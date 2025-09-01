# Real-Time Industry Insight & Strategic Intelligence System

A Python-based system to fetch, process, and analyze real-time data from Twitter (X) for industry insights and strategic intelligence. This project collects tweets based on specific queries and stores them in a structured CSV format for further analysis.

---

## Features

- **Fetches real-time tweets** using Twitter API (X)
- **Stores tweets in CSV** with detailed metadata:
  - `query` – search keyword  
  - `title` – tweet text  
  - `link` – direct tweet URL  
  - `published` – tweet creation timestamp  
  - `fetched_at_utc` – timestamp of fetch
- Handles API rate limits gracefully
- Configurable queries and CSV output
- Easy to extend for multiple queries or analytics

---

## Project Structure

```
Real-Time-Industry-Insight-Strategic-Intelligence-System/
│
├── src/
│   ├── twitter_collect.py   # Main script to fetch tweets
│   ├── config.py            # Reads environment variables like API token
│   └── ...                  # Other helper modules
│
├── .gitignore
├── README.md
├── requirements.txt         # Python dependencies
└── .env.example             # Example env file without real keys
```

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Loki0865/Real-Time-Industry-Insight-Strategic-Intelligence-System.git
   cd Real-Time-Industry-Insight-Strategic-Intelligence-System
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # Activate:
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env    # macOS/Linux
     copy .env.example .env  # Windows
     ```
   - Edit `.env` and add your **Twitter Bearer Token**:
     ```
     TWITTER_BEARER_TOKEN=YOUR_TWITTER_BEARER_TOKEN_HERE
     ```
   > **Note:** `.env` is ignored by GitHub. Never commit real tokens.

---

## Usage

Run the tweet collection script:
```bash
python src/twitter_collect.py
```
- Modify `query` and `max_results` in the script to fetch different topics or number of tweets.
- Tweets will be saved in a CSV file (`tweets.csv`) in the project directory.

Example CSV columns:

| query  | title          | link                                                                             | published            | fetched_at_utc     |
| ------ | -------------- | -------------------------------------------------------------------------------- | -------------------- | ------------------ |
| python | I love Python! | [https://twitter.com/i/web/status/12345](https://twitter.com/i/web/status/12345) | 2025-09-01T14:00:00Z | 2025-09-01T20:30:00Z |

---

## Contributing

- Fork the repository and submit pull requests.
- Ensure **no sensitive keys** are included in contributions.

---

## License

This project is **MIT Licensed** – see `LICENSE` for details.

---

## `.env.example`

Create a file named `.env.example` in the root of your repo:

```
# Twitter API Bearer Token (replace with your own in .env)
TWITTER_BEARER_TOKEN=YOUR_TWITTER_BEARER_TOKEN_HERE
```

- Users can copy `.env.example` to `.env` and add their own keys.
- Keeps your real keys safe while allowing others to run the project.