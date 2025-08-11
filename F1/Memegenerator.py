import requests


# Class for getting meme templates from the Imgflip API.
class MemeGenerator:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.api_url = "https://api.imgflip.com/caption_image"

        # Template ID's for the selected meme templates.
        self.template_ids = {
            "Drake": "181913649",
            "Expanding Brain": "93895088",
            "Woman Yelling Cat": "188390779",
            "Bernie Sanders": "222403160",
            "Surprised Pikachu": "155067746",
            "This is Fine": "55311130",
            "Success Kid": "61544",
            "Distracted Boyfriend": "112126428",
            "Two Buttons": "87743020"
        }

    # Generating a meme using the Imgflip API.
    def generate_meme(self, template_name, captions, max_font_size=50):
        template_id = self.template_ids.get(template_name)
        if not template_id:
            return {"success": False, "error": f"Template '{template_name}' not found"}

        data = {
            'template_id': template_id,
            'username': self.username,
            'password': self.password,
            'max_font_size': str(max_font_size)
        }
        for i, caption in enumerate(captions):
            data[f'boxes[{i}][text]'] = caption
            data[f'boxes[{i}][color]'] = '#ffffff'
            data[f'boxes[{i}][outline_color]'] = '#000000'

        try:
            response = requests.post(self.api_url, data=data)
            result = response.json()

            if result['success']:
                return {
                    'success': True,
                    'url': result['data']['url'],
                    'page_url': result['data']['page_url']
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error_message', 'Unknown error')
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    # Generating memes from news data.
    def generate_memes_from_news(self, news_data):
        results = []

        print(f"News: {news_data['news']}")
        print(f"Sentiment: {news_data['sentiment']}\n")

        for meme in news_data['memes']:
            template = meme['template']
            captions = meme['captions']

            print(f"Generating {template} meme...")
            result = self.generate_meme(template, captions)

            if result['success']:
                print(f"Success! URL: {result['url']}")
                results.append({
                    'template': template,
                    'url': result['url'],
                    'page_url': result['page_url'],
                    'captions': captions
                })
            else:
                print(f"Failed: {result['error']}")
                results.append({
                    'template': template,
                    'error': result['error'],
                    'captions': captions
                })
            print()

        return results

    # Processing a batch of news items and generating memes for each one.
    def process_news_batch(self, news_items):
        all_results = []
        
        for news_item in news_items:
            meme_data = {
                "news": news_item["news"],
                "sentiment": news_item["sentiment"],
                "memes": [{
                    "template": news_item["template"],
                    "captions": news_item["captions"]
                }]
            }
            print("\nProcessing news item:", meme_data["news"][:100], "...")
            results = self.generate_memes_from_news(meme_data)
            all_results.extend(results)
            
        return all_results