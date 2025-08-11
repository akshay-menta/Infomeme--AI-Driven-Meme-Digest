import google.generativeai as genai
import re

# Class for generating prompts for our Gemini model.
class Promptgemini:

    def configure_gemini(api_key):
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')


    # Prompt for the Gemini model.
    def generate_gemini_prompt(self,news_info, emotion, meme_template, num_caption_parts):
        return f"""
        You are an expert meme caption creator. Generate exactly one set of {num_caption_parts}-part captions for a meme based on:

        NEWS: "{news_info}"
        EMOTION: {emotion}
        TEMPLATE: {meme_template}

        Guidelines:
        1. Create {emotion} captions that work with {meme_template}
        2. Use exactly {num_caption_parts} parts
        3. Keep each part short (3-7 words)
        4. Make it relatable to the news
        5. Format as "Part 1: [text]"
        6. Only output one option
        7. Do not include any form of emoji's
        """


    # Extracting caption parts from the Gemini model's response.
    def extract_caption_parts(self,text):
        parts = re.findall(r'Part \d+: (.*)', text)
        cleaned_parts = [part.strip().replace('*', '') for part in parts]
        return cleaned_parts


    # Getting meme captions from the Gemini model.
    def get_meme_captions(self, model, news_info, emotion, meme_template, num_parts):
        prompt = self.generate_gemini_prompt(news_info, emotion, meme_template, num_parts)
        response = model.generate_content(prompt)
        return self.extract_caption_parts(response.text)
