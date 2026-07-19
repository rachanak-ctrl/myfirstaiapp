from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import streamlit as st
import os

os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Create prompt template for generating tweets
tweet_template = """
You are a professional social media manager.
Generate exactly {number} distinct tweets about the topic: {topic}.

Rules:
- Separate each tweet with a unique marker: [TWEET_BREAK]
- Do not number the tweets.
- Do not include any introductory or concluding text.
- Do not wrap the response in JSON or markdown code blocks.
"""

tweet_prompt = PromptTemplate(template=tweet_template, input_variables=['number', 'topic'])

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Create LLM chain using the prompt template and model
tweet_chain = tweet_prompt | gemini_model

st.header("Tweet Generator")
st.subheader("Generate tweets using Generative AI")

topic = st.text_input("Topic")
number = st.number_input("Number of tweets", min_value=1, max_value=10, value=1, step=1)

if st.button("Generate"):
    if not topic.strip():
        st.warning("Please enter a topic first!")
    else:
        with st.spinner("Writing your tweets..."):
            response = tweet_chain.invoke({"number": number, "topic": topic})
            
            # --- Robust fix for the 'list' object error ---
            if isinstance(response.content, list):
                # Safely extract text from Gemini's content blocks if it arrives as a list
                raw_content = "".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in response.content])
            else:
                raw_content = str(response.content)
            
            st.success("Done!")
            
            # Split by the custom marker and clean whitespace
            raw_tweets = raw_content.split("[TWEET_BREAK]")
            cleaned_tweets = [t.strip() for t in raw_tweets if t.strip()]
            
            # Display each tweet cleanly
            for i, tweet in enumerate(cleaned_tweets, 1):
                st.info(f"**Tweet #{i}**\n\n{tweet}")
