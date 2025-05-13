import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
together_api_key = os.getenv('TOGETHER_API_KEY')

# 🌈 Inject custom CSS for styling
st.markdown("""
    <style>
        /* Page background */
        body, .main {
            background-color: #f5f7fa;
        }

        /* Header style */
        h1, h2 {
            color: #4A90E2;
            text-align: center;
        }

        /* Label styling */
        label, .stSelectbox label, .stTextInput label, .stNumberInput label {
            color: #333;
            font-weight: 600;
        }

        /* Input box size and color */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: #ffffff;
            border: 2px solid #4A90E2;
            border-radius: 8px;
            padding: 12px;
            font-size: 18px;
            width: 100% !important;
            box-sizing: border-box;
        }

        /* Button styling */
        .stButton>button {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            padding: 15px 25px;
            margin-top: 10px;
            width: 100%;
            transition: background 0.3s ease;
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #43a047, #388e3c);
        }

        /* Make layout responsive */
        @media only screen and (max-width: 768px) {
            .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
                font-size: 16px;
                padding: 10px;
            }
            .stButton>button {
                font-size: 16px;
                padding: 12px;
            }
        }

        /* Result box */
        .stMarkdown {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# 🧠 Model Providers
providers = {
    "Google Gemini": ["gemini-1.5-flash-latest"],
    "Together (LLaMA 3.3)": ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"]
}

# 🚀 Streamlit UI
st.title("📚 Personalized Learning Plan Generator")
st.subheader("Generate a study plan using Google Gemini or LLaMA-3.3")

# 🎛️ Select Provider
selected_provider = st.selectbox("🌐 Select a Provider", list(providers.keys()))

# 🌍 Language selection
language_select = st.selectbox("🌎 Language", [
    "English", "Hindi", "French", "Spanish", "German", "Chinese", "Arabic", "Gujarati"
])

# 🎯 User Inputs
topic = st.text_input("📘 Enter the Topic You Want to Learn")
days = st.number_input("📅 Number of Days", min_value=1, max_value=365, value=30)
hours_per_day = st.number_input("⏰ Hours of Study Per Day", min_value=1, max_value=12, value=2)

# 🤖 Initialize the selected model
if selected_provider == "Google Gemini":
    model = ChatGoogleGenerativeAI(model=providers[selected_provider][0], google_api_key=google_api_key)
elif selected_provider == "Together (LLaMA 3.3)":
    model = ChatTogether(
        model=providers[selected_provider][0],
        temperature=0.9,
        max_tokens=3000,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )

# 📝 Prompt Template
template = """
Create a detailed and engaging learning plan for learning {topic} in {language}. 
The plan should last for {days} days, with {hours_per_day} hours of study each day.

Please include the following and use appropriate emojis for each section to make it visually engaging:
📅 - For days and scheduling  
📘 - For study activities  
🔗 - For learning resources  
📈 - For progress indicators  
✅ - For checklists or completion  

Make the response friendly, structured, and return it in {language}.
"""


prompt = PromptTemplate(
    template=template,
    input_variables=["topic", "language", "days", "hours_per_day"]
)

# 🔗 Chain prompt and model
learning_plan_chain = prompt | model

# 🧠 Generate Plan
if st.button("🎓 Generate Learning Plan"):
    if not topic:
        st.warning("⚠️ Please enter a topic.")
    else:
        with st.spinner("Generating your personalized learning plan..."):
            try:
                response = learning_plan_chain.invoke({
                    "topic": topic,
                    "language": language_select,
                    "days": days,
                    "hours_per_day": hours_per_day
                })
                st.success("✅ Here's Your Plan:")
                st.markdown(response.content)
            except Exception as e:
                st.error(f"❌ Error: {e}")
