import streamlit as st
from duckduckgo_search import DDGS  # DuckDuckGo Search package
from datetime import datetime

# Configure Streamlit app
st.set_page_config(page_title="Diet & Exercise News Finder", page_icon="🏋️‍♀️")

# Add custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9; /* Neutral light gray background */
    }
    .header-style {
        color: white; /* White title text */
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-size: 24px;
        background-color: #4caf50; /* Green background for the title section */
        padding: 10px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Custom-styled title
st.markdown("<h1 class='header-style'>🏋️‍♀️ Diet & Exercise News Finder with MultiSwarm</h1>", unsafe_allow_html=True)

# Define the Multiswarm architecture
class MultiswarmAgent:
    """Base class for all agents."""
    def __init__(self, name):
        self.name = name

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Execute method must be implemented by subclasses.")

class NewsSearchAgent(MultiswarmAgent):
    """Agent for searching news articles."""
    def __init__(self):
        super().__init__("NewsSearchAgent")

    def execute(self, topic):
        # Modify the search query to include diet and exercise context
        query = f"{topic} related to diet and exercise news {datetime.now().strftime('%Y-%m')}"
        with DDGS() as ddg:
            results = ddg.text(query, max_results=3)
            if results:
                return [
                    {
                        "title": result['title'],
                        "url": result['href'],
                        "summary": result['body']
                    }
                    for result in results
                ]
            return []

class SynthesisAgent(MultiswarmAgent):
    """Agent for synthesizing news articles."""
    def __init__(self):
        super().__init__("SynthesisAgent")

    def execute(self, news_articles):
        if not news_articles:
            return "No news content available to synthesize."

        synthesized_news = []
        for article in news_articles:
            first_sentence = article['summary'].split('.')[0]
            synthesized_news.append(
                f"**{article['title']}**\n{first_sentence}...\n[Read more]({article['url']})"
            )
        return "\n\n".join(synthesized_news)

class SummaryAgent(MultiswarmAgent):
    """Agent for summarizing synthesized news."""
    def __init__(self):
        super().__init__("SummaryAgent")

    def execute(self, synthesized_news, news_articles):
        if not synthesized_news:
            return "No synthesized content available to summarize.", "No summary available."

        # Generate the paragraph summary
        paragraph_summary = ""
        for article in news_articles:
            paragraph_summary += f"{article['summary']} "

        # Return the existing synthesized content and the new summary paragraph
        return synthesized_news, paragraph_summary.strip()

# Orchestrator for the Multiswarm
class NewsProcessor:
    def __init__(self):
        self.agents = {
            "search": NewsSearchAgent(),
            "synthesis": SynthesisAgent(),
            "summary": SummaryAgent()
        }

    def process(self, topic):
        raw_news = self.agents["search"].execute(topic)
        synthesized_news = self.agents["synthesis"].execute(raw_news)
        final_summary, paragraph_summary = self.agents["summary"].execute(synthesized_news, raw_news)
        return raw_news, synthesized_news, final_summary, paragraph_summary

# User Interface
topic = st.text_input("Enter a topic of interest:", value="AI")
if st.button("Find Related News", type="primary"):
    if topic:
        try:
            processor = NewsProcessor()
            raw_news, synthesized_news, final_summary, paragraph_summary = processor.process(topic)

            # Display the results
            st.header(f"📝 News Summary: {topic}")
            st.markdown(final_summary)

            st.header("📊 Consolidated Summary")
            st.write(paragraph_summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter a topic!")
