from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the API key from the environment
api_key = os.getenv('GOOGLE_API_KEY')

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

# Define a prompt template for generating tweets
tweet_template = "Give me {number} tweets on {topic}"
tweet_prompt = PromptTemplate(template=tweet_template, input_variables=["number", "topic"])

# Combine prompt and model into a chain
tweet_chain = tweet_prompt | gemini_model

# Streamlit UI
st.header("Generator - AI")
st.subheader("Generate sentences using Generative AI")

topic = st.text_input("Enter a topic")
number = st.number_input("Number of sentence", min_value=1, max_value=10, value=1, step=1)

if st.button("Generate"):
    if topic.strip() == "":
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Generating sentence..."):
            tweets = tweet_chain.invoke({"number": number, "topic": topic})
            st.success("Here are your sentences:")
            st.write(tweets.content)
