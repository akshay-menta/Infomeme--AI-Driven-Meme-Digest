# InfoMeme: AI-Powered News to Meme Generator 

A sophisticated AI system that transforms news articles into contextually relevant and engaging memes by leveraging state-of-the-art language models, emotion analysis and fake news detection.

## 🌟 Features

- **News Article Processing**: Fetches and processes news articles from various categories using NewsAPI
- **Article Summarization**: Leverages T5-small model to create concise, informative summaries of news articles
- **Fake News Detection**: Employs fine-tuned BERT/RoBERTa models to filter out unreliable news
- **Emotion Analysis**: Utilizes fine-tuned RoBERTa model for multi-label emotion classification
- **Meme Generation Pipeline**:
  - Automatically fetches news articles via NewsAPI or processes user-provided URLs, converting them into concise, meme-friendly summaries using T5-small model
  - Analyzes emotional context of the summary
  - Creates contextually appropriate meme captions using Google's Gemini Pro
  - Matches sentiment with suitable meme templates
  - Generates final meme using ImgFlip API
- **Streamlit Interface**: User-friendly web interface for easy interaction

## 🛠️ Technical Architecture

### Core Components (F1/)

- `app.py`: Main Streamlit application with responsive UI
- `NewsToMemeGenerator.py`: Core orchestrator integrating all components
- `FakeNewsDetector.py`: Fake news classification implementation
- `EmotionAnalyzer.py`: Multi-label emotion classification
- `Memegenerator.py`: Meme creation using ImgFlip API
- `NewsAPIClient.py`: News article fetching and processing
- `Promptgemini.py`: Gemini flash-1.5 API integration for caption generation
- `URLContentSummarizer.py`: Article content extraction and summarization

### Model Training (F2/)

#### Fake News Detection
- `Bert_base_fakenews.py`: BERT base model fine-tuning
- `Roberta_base_fakenews.py`: RoBERTa base model fine-tuning

#### Sentiment Analysis
- `Roberta_base_emotion.py`: RoBERTa model for emotion classification
- `T5_base_emotion.py`: T5 model experimentation

## 🚀 Getting Started

### Prerequisites

```bash
python -v 3.8+
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with the following:

```env
NEWS_API_KEY=your_news_api_key
GEMINI_API_KEY=your_gemini_api_key
IMGFLIP_USERNAME=your_imgflip_username
IMGFLIP_PASSWORD=your_imgflip_password
```

### Running the Application

```bash
cd F1
streamlit run app.py
```

## 🔧 Model Details

### Fake News Detection
- Architecture: RoBERTa Base (chosen after comparing with BERT)
- Dataset: LIAR-2 dataset
- Metrics: F1-score, Precision, Recall
- Classes: pants-fire, false, barely-true, half-true, mostly-true, true

### Emotion Analysis
- Architecture: RoBERTa Base (chosen after comparing with T5)
- Dataset: GoEmotions
- Type: Multi-label classification
- Emotions: 27 fine-grained emotions
- Metrics: Micro-F1, Macro-F1, Hamming Loss

### Article Summarization
- Architecture: T5-small
- Dataset: XSum (Extreme Summarization)
- Training Configuration:
  - Max Input Length: 512 tokens
  - Max Summary Length: 64 tokens
  - Learning Rate: 5e-5
  - Batch Size: 4 (with gradient accumulation)
- Metrics: ROUGE-1, ROUGE-2, ROUGE-L scores
- Features:
  - Abstractive summarization
  - Efficient text compression
  - Context-aware summary generation

## 📊 Performance

- Fake News Detection: 
  - Accuracy: ~76%
  - Macro F1: ~72%
  - Weighted F1: ~75%

- Emotion Analysis:
  - Micro F1: ~71%
  - Macro F1: ~65%
  - Hamming Loss: ~0.08

## 🎯 Use Cases

1. **News Media**: Generate engaging social media content
2. **Content Marketing**: Create viral-worthy memes from company updates
3. **Educational**: Transform complex news into digestible memes
4. **Social Commentary**: Express opinions through contextual humor

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NewsAPI for news article access
- Google's Gemini Pro for text generation
- ImgFlip API for meme creation
- HuggingFace for transformer models and datasets
