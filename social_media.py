from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import Together
from langchain_core.prompts import PromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys
google_api_key = os.getenv('GOOGLE_API_KEY')
together_api_key = os.getenv('TOGETHER_API_KEY')

# Define available providers
providers = ["Google Gemini", "Together AI"]

# Streamlit UI setup
st.set_page_config(
    page_title="Social Media Post Generator",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Social Media Post Generator")
st.markdown("Generate engaging social media posts with AI")

# Sidebar: model selection
with st.sidebar:
    st.header("⚙️ Settings")
    selected_provider = st.selectbox("Select AI Provider", providers)

    # Initialize model
    model = None
    if selected_provider == "Google Gemini":
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key)
    elif selected_provider == "Together AI":
        model = Together(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            together_api_key=together_api_key,
            temperature=0.7,
            max_tokens=1000
        )

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    platform = st.selectbox(
        "Select Platform",
        ["Twitter", "LinkedIn", "Facebook", "Instagram", "TikTok", "YouTube"]
    )

    topic = st.text_area(
        "What's your post about?",
        placeholder="Enter the main topic or message of your post..."
    )

    tone = st.selectbox(
        "Select Tone",
        ["Professional", "Casual", "Funny", "Inspirational", "Educational",
         "Enthusiastic", "Witty", "Friendly", "Authoritative", "Humorous",
         "Motivational", "Conversational", "Sarcastic", "Playful", "Serious"],
        help="Choose the tone that best matches your brand voice"
    )

    audience_options = [
        "General Public", "Professionals", "Students", "Entrepreneurs", "Tech Enthusiasts",
        "Creative Artists", "Health & Fitness", "Food Lovers", "Travelers", "Parents",
        "Gamers", "Fashion Enthusiasts", "Business Owners", "Educators", "Sports Fans", "Other"
    ]

    selected_audiences = st.multiselect(
        "Select Target Audience",
        audience_options,
        default=["General Public"]
    )

    if "Other" in selected_audiences:
        custom_audience = st.text_input("Enter your custom audience", placeholder="e.g., Pet Owners, Music Lovers, etc.")
        selected_audiences.remove("Other")
        if custom_audience:
            selected_audiences.append(custom_audience)

with col2:
    language_select = st.selectbox(
        "Select Language",
        ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Japanese", "Korean", "Chinese"]
    )

# Prompt template
post_template = """
Create an engaging social media post for {platform} about {topic}.
The post should be in {language} language and have a {tone} tone.
Target audience: {targeted_audience}

Guidelines:
- Keep it concise and engaging
- Make it platform-appropriate
- Ensure it's authentic and relatable
- Add a call-to-action if appropriate
- Tailor the content to the selected target audience
- Maintain a consistent {tone} tone throughout the post

Hashtag Requirements:
- Generate 5-7 relevant hashtags
- Include a mix of:
  * Topic-specific hashtags
  * Industry-related hashtags
  * Trending hashtags (if applicable)
  * Location-based hashtags (if relevant)
  * Brand or campaign hashtags (if applicable)
- Format hashtags without spaces
- Use camelCase for multi-word hashtags
- Keep hashtags concise and memorable

Please format the post with:
1. Main content
2. Line break
3. Hashtags section starting with "#"
4. Any platform-specific elements

Example format:
[Your engaging post content here]

#Hashtag1 #Hashtag2 #Hashtag3 #Hashtag4 #Hashtag5
"""

post_prompt = PromptTemplate(
    template=post_template,
    input_variables=["platform", "topic", "language", "tone", "targeted_audience"]
)

post_chain = post_prompt | model

# State control
if 'generating' not in st.session_state:
    st.session_state.generating = False
if 'first_click' not in st.session_state:
    st.session_state.first_click = True
if 'trigger_generate' not in st.session_state:
    st.session_state.trigger_generate = False

# UI separator
st.markdown("---")

# Button logic
if st.session_state.first_click and st.session_state.generating:
    # Spinner replaces button on first click
    with st.spinner("Generating your post..."):
        try:
            response = post_chain.invoke({
                "platform": platform,
                "language": language_select,
                "topic": topic,
                "tone": tone,
                "targeted_audience": ", ".join(selected_audiences)
            })
            st.success("✅ Here's your generated post:")
            if selected_provider == "Together AI":
                st.markdown(response)
            else:
                st.markdown(response.content)
            st.caption("📌 Hashtags are auto-generated.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            st.session_state.generating = False
            st.session_state.first_click = False
else:
    if st.button("🚀 Generate Post", use_container_width=True, disabled=st.session_state.generating):
        if not topic:
            st.warning("⚠️ Please enter a topic for your post.")
        else:
            st.session_state.generating = True
            st.session_state.trigger_generate = True
            st.rerun()

# Handle future clicks (without spinner)
if st.session_state.trigger_generate and not st.session_state.first_click:
    try:
        response = post_chain.invoke({
            "platform": platform,
            "language": language_select,
            "topic": topic,
            "tone": tone,
            "targeted_audience": ", ".join(selected_audiences)
        })
        st.success("✅ Here's your generated post:")
        if selected_provider == "Together AI":
            st.markdown(response)
        else:
            st.markdown(response.content)
        st.caption("📌 Hashtags are auto-generated.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        st.session_state.generating = False
        st.session_state.trigger_generate = False

# Footer
st.markdown("---")
st.caption("Thank you for using my app!❤️")
