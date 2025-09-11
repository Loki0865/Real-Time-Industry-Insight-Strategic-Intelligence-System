# Real-Time Industry Insight Strategic Intelligence System

A comprehensive real-time data collection and sentiment analysis system that gathers industry insights from multiple news sources and provides strategic intelligence through AI-powered sentiment analysis.

## 🚀 Features

- **Multi-Source News Collection**: Fetches real-time news from NewsAPI and Google News RSS feeds
- **AI-Powered Sentiment Analysis**: Uses Google's Gemini AI model for advanced sentiment classification
- **Multi-Topic Monitoring**: Tracks multiple industry topics simultaneously (AI, Technology, Market trends, etc.)
- **Data Processing Pipeline**: Automated data cleaning, deduplication, and processing
- **CSV Export**: Processed data saved in structured CSV format for further analysis
- **Batch Processing**: Efficient batch processing for handling large datasets
- **Configurable Topics**: Easily customizable topics and data sources

## 📋 Prerequisites

- Python 3.10 or higher
- NewsAPI account and API key
- Google Cloud account with Generative AI API access
- Virtual environment (recommended)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Loki0865/Real-Time-Industry-Insight-Strategic-Intelligence-System.git
   cd Real-Time-Industry-Insight-Strategic-Intelligence-System
   ```

2. **Set up virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd "Real time data fetcher"
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

1. **Copy the environment template**
   ```bash
   cd "Real time data fetcher"
   cp .env.example .env
   ```

2. **Configure API keys in `.env` file**
   ```env
   # NewsAPI key from https://newsapi.org/
   NEWS_API_KEY=your_newsapi_key_here
   
   # Google AI API key from Google Cloud Console
   GOOGLE_API_KEY=your_google_ai_api_key_here
   
   # Optional: Twitter Bearer Token for future Twitter integration
   TWITTER_BEARER_TOKEN=your_twitter_token_here
   ```

### API Setup Instructions

#### NewsAPI Setup
1. Visit [NewsAPI.org](https://newsapi.org/)
2. Create a free account
3. Get your API key from the dashboard
4. Add it to your `.env` file

#### Google AI Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Generative AI API
4. Create credentials (API key)
5. Add it to your `.env` file

## 🚀 Usage

### Basic Usage

```bash
cd "Real time data fetcher"
python src/Sentiment_analysis.py
```

### Custom Topics

You can modify the topics in the script or create your own runner:

```python
from src.Sentiment_analysis import run_pipeline

# Custom topics
topics = ["Artificial Intelligence", "Blockchain", "Cybersecurity", "Market Analysis"]
run_pipeline(topics)
```

### Output

The system generates two types of output files:

1. **Raw Data**: `data/raw/news_multi_YYYYMMDD_HHMMSS.csv`
   - Contains unprocessed news articles with metadata

2. **Processed Data**: `data/processed/news_with_sentiment_multi_YYYYMMDD_HHMMSS.csv`
   - Contains cleaned data with sentiment analysis results

## 📁 Project Structure

```
Real-Time-Industry-Insight-Strategic-Intelligence-System/
├── Real time data fetcher/
│   ├── src/
│   │   ├── Sentiment_analysis.py    # Main processing script
│   │   └── data/                    # Generated data files
│   ├── data/
│   │   ├── raw/                     # Raw news data
│   │   └── processed/               # Processed data with sentiment
│   ├── requirements.txt             # Python dependencies
│   ├── .env.example                # Environment variables template
│   └── README.md                   # Week 1 setup guide
├── .gitignore
└── README.md                       # This file
```

## 🔧 System Components

### Data Collection
- **NewsAPI Integration**: Fetches articles using keyword queries
- **Google News RSS**: Collects additional news sources via RSS feeds
- **Data Deduplication**: Removes duplicate articles across sources

### Text Processing
- **Text Cleaning**: Removes URLs, hashtags, emojis, and special characters
- **Content Truncation**: Limits text to 500 characters for processing efficiency
- **Data Standardization**: Ensures consistent formatting across sources

### Sentiment Analysis
- **Gemini AI Integration**: Uses Google's advanced language model
- **Batch Processing**: Processes multiple articles efficiently
- **Sentiment Scoring**: Provides both sentiment labels and confidence scores

## 🎯 Use Cases

- **Market Intelligence**: Monitor industry trends and sentiment
- **Brand Monitoring**: Track news sentiment about specific companies or products
- **Investment Research**: Analyze market sentiment for investment decisions
- **Competitive Analysis**: Monitor competitor mentions and industry developments
- **Risk Assessment**: Identify potential risks through sentiment analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Development Guidelines

- Follow PEP 8 coding standards
- Add appropriate error handling
- Include docstrings for functions
- Test with sample data before processing large datasets
- Monitor API rate limits

## ⚠️ Important Notes

- **API Limits**: Be aware of rate limits for NewsAPI and Google AI API
- **Data Privacy**: Ensure compliance with data privacy regulations
- **Cost Management**: Monitor Google AI API usage to avoid unexpected charges
- **Data Storage**: Raw and processed data files can grow large over time

## 🔐 Security

- Never commit API keys to version control
- Use environment variables for all sensitive configuration
- Keep your `.env` file in `.gitignore`
- Regularly rotate API keys

## 📈 Future Enhancements

- Twitter/X integration for social media sentiment
- Real-time dashboard for live monitoring
- Advanced analytics and reporting features
- Machine learning model training for custom sentiment analysis
- Multi-language support
- Database integration for persistent storage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainer.

---

**Note**: This system is designed for research and analytical purposes. Ensure compliance with all relevant APIs' terms of service and data usage policies.