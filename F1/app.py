import streamlit as st
import json
import os
import requests
from dotenv import load_dotenv
from NewsToMemeGenerator import NewsToMemeGenerator
load_dotenv()

st.set_page_config(
    page_title="News to Meme Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-size: 18px;
    }

    .stMarkdown p, .stMarkdown {
        font-size: 18px !important;
        line-height: 1.6;
    }

    h1 {
        font-size: 48px !important;
    }

    h2 {
        font-size: 36px !important;
    }

    h3 {
        font-size: 28px !important;
    }

    .stButton > button {
        font-size: 18px !important;
        padding: 0.75rem 1.5rem !important;
        width: 100%;
        margin-top: 1rem;
    }

    .stSelectbox label, .stTextInput label {
        font-size: 20px !important;
        font-weight: 600;
    }

    .stSelectbox > div > div, .stTextInput > div > div > input {
        font-size: 18px !important;
    }

    .stRadio > label {
        font-size: 20px !important;
        font-weight: 600;
    }

    .stRadio > div > label > div {
        font-size: 18px !important;
    }

    .streamlit-expanderHeader {
        font-size: 20px !important;
        font-weight: 600;
    }

    .stCodeBlock > div > pre {
        font-size: 16px !important;
    }

    /* Style improvements */
    .main {
        padding-top: 2rem;
    }

    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .meme-container {
        text-align: center;
        margin: 20px 0;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    .sentiment-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
        font-size: 18px !important;
    }

    .css-1d391kg p {
        font-size: 16px !important;
    }

    .stAlert > div {
        font-size: 18px !important;
    }

    [data-testid="metric-container"] > div > div {
        font-size: 20px !important;
    }

    .mode-container {
        background: transparent;
        border-radius: 20px;
        padding: 2rem 2rem 3rem 2rem;
        margin: 2rem 0;
        position: relative;
    }

    .toggle-container {
        background-color: rgba(240, 242, 246, 0.1);
        border-radius: 35px;
        padding: 6px;
        display: flex;
        gap: 6px;
        margin-bottom: 2rem;
        width: 100%;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .toggle-container [data-testid="column"] > div > div > div > button {
        border-radius: 30px !important;
        border: none !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        padding: 14px 24px !important;
        transition: all 0.3s ease !important;
        height: 56px !important;
    }

    .toggle-container .stButton > button[kind="primary"] {
        background-color: #4CAF50 !important;
        color: white !important;
        box-shadow: 0 2px 10px rgba(76, 175, 80, 0.3) !important;
    }

    .toggle-container .stButton > button[kind="primary"]:hover {
        background-color: #45a049 !important;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4) !important;
    }

    .toggle-container .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #666 !important;
        border: none !important;
    }

    .toggle-container .stButton > button[kind="secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.8) !important;
        color: #333 !important;
    }

    .content-area {
        padding: 0 1rem;
    }

    .toggle-container .stButton {
        margin: 0 !important;
    }

    .toggle-container .stButton > button {
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'mode' not in st.session_state:
    st.session_state.mode = "Auto Generate"

st.title("Infomeme : News to Meme Generator")
st.markdown("### AI-Driven Meme Digest ✨")

with st.sidebar:
    st.header("⚙️ Configuration")

    use_env = st.checkbox("Use environment variables", value=True)

    if use_env:
        news_api_key = os.getenv("NEWS_API_KEY", "")
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        imgflip_username = os.getenv("IMGFLIP_USERNAME", "")
        imgflip_password = os.getenv("IMGFLIP_PASSWORD", "")

        if news_api_key:
            st.text(f"News API Key: {'*' * 10}")
        if gemini_api_key:
            st.text(f"Gemini API Key: {'*' * 10}")
        if imgflip_username:
            st.text(f"Imgflip Username: {imgflip_username}")
        if imgflip_password:
            st.text(f"Imgflip Password: {'*' * 10}")
    else:
        news_api_key = st.text_input("News API Key", type="password")
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        imgflip_username = st.text_input("Imgflip Username")
        imgflip_password = st.text_input("Imgflip Password", type="password")

    if st.button("Initialize Generator", type="primary"):
        if all([news_api_key, gemini_api_key, imgflip_username, imgflip_password]):
            try:
                with st.spinner("Initializing generator..."):
                    st.session_state.generator = NewsToMemeGenerator(
                        news_api_key,
                        gemini_api_key,
                        imgflip_username,
                        imgflip_password
                    )
                st.success("Generator initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing: {str(e)}")
        else:
            st.error("Please fill in all API credentials!")

    st.markdown("---")
    if st.session_state.generator:
        st.success("Generator is ready!")
    else:
        st.warning("Generator not initialized")

if st.session_state.generator is None:
    st.info("Please configure your API keys in the sidebar to get started!")
    st.stop()


st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
cols = st.columns(2)
with cols[0]:
    if st.button("🤖 Auto Generate", use_container_width=True, type="primary" if st.session_state.mode == "Auto Generate" else "secondary"):
        st.session_state.mode = "Auto Generate"
        st.rerun()
with cols[1]:
    if st.button("✍️ Manual", use_container_width=True, type="primary" if st.session_state.mode == "Manual" else "secondary"):
        st.session_state.mode = "Manual"
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)  # Close toggle container

# Content area
# Auto Generate Mode
if st.session_state.mode == "Auto Generate":
    st.header("🤖 Auto Generate Memes from News")

    col1, col2 = st.columns(2)

    with col1:
        # Topic dropdown
        topic = st.selectbox(
            "Select News Topic",
            options=[
                "technology",
                "business",
                "entertainment",
                "general",
                "health",
                "science",
                "sports"
            ],
            help="Choose the news category to fetch articles from"
        )

    with col2:
        # Number of memes
        num_memes = st.selectbox(
            "Number of Memes to Generate",
            options=[1, 2, 3, 4, 5],
            help="Select how many memes you want to generate"
        )

    # Generate button
    if st.button("🎨 Generate Memes", type="primary", use_container_width=True):
        with st.spinner(f"🔄 Fetching {topic} news and generating {num_memes} meme(s)..."):
            try:
                # Call the extractnewsDataIntoMemes method
                result = st.session_state.generator.extract_news_data_into_memes(topic, num_memes)

                if result and result.get("news_data"):
                    st.session_state.results = result
                    successful_count = sum(
                        1 for item in result["news_data"]
                        if item.get("meme_results") and any('url' in r for r in item["meme_results"])
                    )
                    st.success(f" Successfully generated {successful_count} meme(s)!")
                else:
                    st.error(" No memes were generated. Please try again.")
                    st.session_state.results = None

            except Exception as e:
                st.error(f" An error occurred: {str(e)}")
                st.session_state.results = None

else:
    st.header("✍️ Generate Meme from URL")

    url = st.text_input(
        "Enter Article URL",
        placeholder="https://example.com/news-article",
        help="Paste the URL of the news article you want to convert to a meme"
    )

    if st.button("🎨 Generate Meme", type="primary", disabled=not url, use_container_width=True):
        with st.spinner("🔄 Processing URL and generating meme..."):
            try:
                # Call the process_url_input method
                result = st.session_state.generator.process_url_input(
                    url,
                    st.session_state.generator.emotion_analyzer,
                    st.session_state.generator.meme_generator,
                    st.session_state.generator.gemini_model
                )

                if result and result.get("news_data"):
                    st.session_state.results = result
                    st.success(" Meme generated successfully!")
                else:
                    st.error(" Failed to generate meme from URL")
                    st.session_state.results = None

            except Exception as e:
                st.error(f" An error occurred: {str(e)}")
                st.session_state.results = None


def get_image_download_link(img_url):
    """Download image from URL and return as bytes for download button"""
    try:
        response = requests.get(img_url)
        if response.status_code == 200:
            return response.content
    except:
        return None
    return None


# Display results
if st.session_state.results and st.session_state.results.get("news_data"):
    st.markdown("---")
    st.header("📊 Results")

    for idx, news_item in enumerate(st.session_state.results["news_data"], 1):
        sentiment = news_item.get('sentiment', 'Unknown')

        sentiment_colors = {
            'admiration': '#9c27b0',
            'amusement': '#ff9800',
            'anger': '#dc3545',
            'annoyance': '#fd7e14',
            'approval': '#4caf50',
            'caring': '#e91e63',
            'confusion': '#9e9e9e',
            'curiosity': '#00bcd4',
            'desire': '#d32f2f',
            'disappointment': '#795548',
            'disapproval': '#f44336',
            'disgust': '#4a148c',
            'embarrassment': '#ff5722',
            'excitement': '#ffc107',
            'fear': '#37474f',
            'gratitude': '#2e7d32',
            'grief': '#424242',
            'joy': '#28a745',
            'love': '#c2185b',
            'nervousness': '#607d8b',
            'optimism': '#ffeb3b',
            'pride': '#673ab7',
            'realization': '#3f51b5',
            'relief': '#8bc34a',
            'remorse': '#5d4037',
            'sadness': '#1976d2',
            'surprise': '#17a2b8',
            'neutral': '#6c757d'
        }

        sentiment_color = sentiment_colors.get(sentiment.lower(), '#6c757d')

        with st.expander(f"📰 News Item {idx}", expanded=True):
            st.markdown(
                f'<div class="sentiment-badge" style="background-color: {sentiment_color}; color: white;">'
                f'{sentiment.upper()}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.subheader("📄 News Content")
            with st.container():
                st.write(news_item["news"])

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Template:** {news_item.get('template', 'Unknown')}")
            with col2:
                st.info(f"**Parts:** {news_item.get('parts', 'Unknown')}")

            if news_item.get("captions"):
                st.subheader("💬 Generated Captions")
                for i, caption in enumerate(news_item["captions"], 1):
                    st.write(f"**Caption {i}:** {caption}")

            if news_item.get("meme_results") and len(news_item["meme_results"]) > 0:
                meme_result = news_item["meme_results"][0]
                if meme_result.get("url"):
                    st.subheader("🖼️ Generated Meme")

                    # Center the meme image
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(meme_result["url"], use_container_width=True)

                        # Download meme button
                        meme_img_data = get_image_download_link(
                            meme_result["url"]
                        )

                        if meme_img_data:
                            st.download_button(
                                label="📥 Download Meme",
                                data=meme_img_data,
                                file_name=f"news_meme_{idx}_{sentiment.lower()}.jpg",
                                mime="image/jpeg",
                                use_container_width=True
                            )
                else:
                    st.warning(" Meme URL not available")
            else:
                st.warning(" No meme was generated for this item")

    st.markdown("---")
    with st.expander("📋 View JSON Output", expanded=False):
        json_output = json.dumps(st.session_state.results, indent=2)
        st.code(json_output, language="json")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.results = None
            st.rerun()

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p style='font-size: 20px;'>Transforming everyday news into shareable content instantly</p>
        <p style='font-size: 16px;'>Thank you for using the application</p>
    </div>
    """,
    unsafe_allow_html=True
)