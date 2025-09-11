# Real-Time Industry Insight Strategic Intelligence System

A comprehensive AI-powered system for collecting, processing, and analyzing real-time industry news and insights. This system provides strategic intelligence through automated news collection from multiple sources and advanced sentiment analysis using Google's Gemini AI model.

## 🌟 Features

- **Multi-Source News Collection**: Automatically fetch news from NewsAPI and Google News RSS feeds
- **AI-Powered Sentiment Analysis**: Leverage Google's Gemini 1.5 Flash model for accurate sentiment classification
- **Real-Time Processing**: Collect and process industry news in real-time with timestamped data
- **Batch Processing**: Efficient sentiment analysis with configurable batch sizes
- **Topic Flexibility**: Support for multiple industry topics (AI, Technology, Market trends, etc.)
- **Data Deduplication**: Intelligent removal of duplicate articles
- **Structured Data Output**: Clean, processed data exported to CSV format
- **Text Cleaning**: Advanced text preprocessing with emoji removal and character filtering

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- VS Code with Python extension (recommended)
- API keys for NewsAPI and Google Generative AI

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Loki0865/Real-Time-Industry-Insight-Strategic-Intelligence-System.git
   cd Real-Time-Industry-Insight-Strategic-Intelligence-System
   ```

2. **Navigate to the main application directory**
   ```bash
   cd "Real time data fetcher"
   ```

3. **Create and activate a virtual environment**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   
   # Windows (CMD)
   python -m venv .venv
   .\.venv\Scripts\activate.bat
   
   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your API keys
   NEWS_API_KEY=your_newsapi_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   TWITTER_BEARER_TOKEN=your_twitter_token_here  # Optional
   ```

### API Key Setup

#### NewsAPI Key
1. Visit [NewsAPI.org](https://newsapi.org/)
2. Sign up for a free account
3. Copy your API key to the `.env` file

#### Google Generative AI Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy your API key to the `.env` file

## 📖 Usage

### Basic Usage

Run the sentiment analysis pipeline with default topics:

```bash
python src/Sentiment_analysis.py
```

This will:
1. Fetch news articles for predefined topics (AI, Technology, Trending, Market, AI Trends)
2. Clean and preprocess the text data
3. Perform sentiment analysis using Google's Gemini model
4. Save results to `data/processed/` directory

### Custom Topics

Modify the topics in `src/Sentiment_analysis.py`:

```python
if __name__ == "__main__":
    topics_to_fetch = ["Your Custom Topic", "Another Topic", "Market Analysis"]
    run_pipeline(topics_to_fetch)
```

### Output Files

The system generates two types of output files:

- **Raw Data**: `data/raw/news_multi_YYYYMMDD_HHMMSS.csv`
  - Original article data from all sources
  - Includes title, description, URL, source, timestamp

- **Processed Data**: `data/processed/news_with_sentiment_multi_YYYYMMDD_HHMMSS.csv`
  - Cleaned text data with sentiment analysis results
  - Includes sentiment labels (Positive/Negative/Neutral) and confidence scores

## 🏗️ Project Structure

```
Real-Time-Industry-Insight-Strategic-Intelligence-System/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
└── Real time data fetcher/            # Main application directory
    ├── README.md                      # Setup and usage instructions
    ├── requirements.txt               # Python dependencies
    ├── .env.example                   # Environment variables template
    ├── data/                          # Data storage directory
    │   ├── raw/                       # Raw news data
    │   └── processed/                 # Processed data with sentiment
    └── src/                           # Source code
        └── Sentiment_analysis.py     # Main application logic
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NEWS_API_KEY` | API key for NewsAPI.org | Yes |
| `GOOGLE_API_KEY` | Google Generative AI API key | Yes |
| `TWITTER_BEARER_TOKEN` | Twitter API bearer token | Optional |

### Customizable Parameters

- **Batch Size**: Adjust sentiment analysis batch size (default: 10)
- **Max Articles**: Configure maximum articles per topic per source
- **Language**: Set news language (default: English)
- **Topics**: Define custom industry topics for monitoring

## 📊 Data Schema

### Raw Data Fields
- `title`: Article headline
- `description`: Article summary
- `content`: Full article content (when available)
- `publishedAt`: Publication timestamp
- `url`: Article URL
- `source`: News source name
- `topic`: Search topic used

### Processed Data Fields
All raw data fields plus:
- `clean_text`: Preprocessed text for analysis
- `sentiment_gemini`: Sentiment label (Positive/Negative/Neutral)
- `sentiment_score`: Confidence score (-1.0 to 1.0)

## 🤖 AI Model Details

The system uses Google's **Gemini 1.5 Flash** model for sentiment analysis:

- **Model**: `gemini-1.5-flash`
- **Output**: Structured JSON with sentiment labels and confidence scores
- **Processing**: Batch processing for efficiency
- **Validation**: Built-in error handling and data validation

## 🛠️ Development

### Running in VS Code

1. Open the project folder in VS Code
2. Select the Python interpreter from your virtual environment
3. Use the integrated terminal to run commands
4. Leverage VS Code's Python debugging features

### Adding New Features

The modular design makes it easy to extend functionality:

- **New Data Sources**: Add functions similar to `fetch_news()` and `fetch_google_news()`
- **Additional Analysis**: Extend the processing pipeline in `run_pipeline()`
- **Custom Models**: Replace or supplement the Gemini model integration

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Useful Links

- [NewsAPI Documentation](https://newsapi.org/docs)
- [Google Generative AI Documentation](https://ai.google.dev/docs)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)

## 🆘 Support

If you encounter any issues or have questions:

1. Check the existing [Issues](https://github.com/Loki0865/Real-Time-Industry-Insight-Strategic-Intelligence-System/issues)
2. Create a new issue with detailed information
3. Include error messages and system information

---

**Built with ❤️ for strategic intelligence and data-driven insights**