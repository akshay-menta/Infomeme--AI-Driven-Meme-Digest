import warnings

warnings.filterwarnings('ignore', message='`resume_download` is deprecated')

from constants import Emotion_Template_Mapping, Last_Month_News
from NewsAPIClient import NewsAPIClient
from FakeNewsDetector import FakeNewsDetector
from EmotionAnalyzer import EmotionAnalyzer
from Promptgemini import Promptgemini
from Memegenerator import MemeGenerator
from URLContentSummarizer import URLContentSummarizer

# Main class for generating memes from news.
class NewsToMemeGenerator:
    def __init__(self, news_api_key, gemini_api_key, imgflip_username, imgflip_password):
        self.news_client = NewsAPIClient(news_api_key)
        self.fake_news_detector = FakeNewsDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.meme_generator = MemeGenerator(imgflip_username, imgflip_password)
        self.gemini_model = Promptgemini.configure_gemini(gemini_api_key)
        self.prompt_gemini = Promptgemini()
        self.url_summarizer = URLContentSummarizer()


    # Generating a meme for a single news item.
    def generate_meme_for_news_item(self,news_item):
        meme_data = {
            "news": news_item["news"],
            "sentiment": news_item["sentiment"],
            "memes": [{
                "template": news_item["template"],
                "captions": news_item["captions"]
            }]
        }

        print(f"\nGenerating meme for: {news_item['news'][:100]}...")
        results = self.meme_generator.generate_memes_from_news(meme_data)
        return results


    # Processing a single URL.
    def process_url_input(self, url, emotion_analyzer, meme_generator, gemini_model):
        output_data = {"news_data": []}

        try:
            # Getting summary from URL
            summary = self.url_summarizer.process_url(url)
            print(f"\nGenerated summary from URL:\n{summary}\n")

            # Analyzing emotion of the summary
            emotion_results = emotion_analyzer.predict_emotion([summary])
            simplified_data = EmotionAnalyzer.prepare_for_meme_generation(emotion_results)

            if not simplified_data:
                raise Exception("Could not analyze emotion from the summary")

            emotion_result = simplified_data[0]  

            # Getting template and parts based on sentiment
            sentiment_lower = emotion_result["sentiment"].lower()
            template_info = Emotion_Template_Mapping.get(sentiment_lower, {"template": "Drake", "parts": 2})

            print(f"\nDetected emotion: {emotion_result['sentiment']}")
            print(f"Using template: {template_info['template']}")

            # Creating news item with summary
            news_item = {
                "news": summary,
                "sentiment": emotion_result["sentiment"],
                "template": template_info["template"],
                "parts": template_info["parts"]
            }

            # Generating captions
            prompt_gemini = Promptgemini()
            captions = prompt_gemini.get_meme_captions(
                gemini_model,
                news_item["news"],
                news_item["sentiment"],
                news_item["template"],
                num_parts=news_item["parts"]
            )
            news_item["captions"] = captions

            print("\nGenerated Captions:")
            for i, caption in enumerate(captions, 1):
                print(f"Caption {i}: {caption}")

            # Generating final meme
            meme_results = self.generate_meme_for_news_item(news_item)
            news_item["meme_results"] = meme_results
            output_data["news_data"].append(news_item)

            return output_data

        except Exception as e:
            print(f"Error processing URL: {str(e)}")
            return None


    # Extracting news data into memes.
    def extract_news_data_into_memes(self, topic, n) :
        output_data = {"news_data": []}

        news_articles = self.news_client.get_news_by_timeframe(
            timeframe_hours=Last_Month_News, # Optional: change this to Last_Week_News or Last_Year_News for different timeframes.
            category= topic
        )

        if not news_articles:
            print("No news articles to process. Exiting.")
            return []

        # Detecting fake news
        fake_news_results = self.fake_news_detector.predict_batch(news_articles)
        true_news = self.fake_news_detector.filter_true_news(n,fake_news_results)

        if not true_news:
            print("No true news articles found. Exiting.")
            return []
        emotion_results = self.emotion_analyzer.predict_emotion(true_news)
        simplified_data = EmotionAnalyzer.prepare_for_meme_generation(emotion_results)

        # Processing each news item
        prompt_gemini = Promptgemini()
        for item in simplified_data:
            sentiment_lower = item["sentiment"].lower()
            template_info = Emotion_Template_Mapping.get(sentiment_lower, {"template": "Drake", "parts": 2})

            news_item = {
                "news": item["news"],
                "sentiment": item["sentiment"],
                "template": template_info["template"],
                "parts": template_info["parts"]
            }

            captions = prompt_gemini.get_meme_captions(
                self.gemini_model,
                news_item["news"],
                news_item["sentiment"],
                news_item["template"],
                num_parts=news_item["parts"]
            )
            news_item["captions"] = captions

            meme_results = self.generate_meme_for_news_item(news_item)
            news_item["meme_results"] = meme_results
            output_data["news_data"].append(news_item)

        return output_data