import streamlit as st
import requests
from llama_index.llms.ollama import Ollama
from pathlib import Path
from datetime import datetime
import qdrant_client
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

# Function to get weather data from OpenWeatherMap
def get_weather(city):
    api_key = "7724b1e7e80c33e489ff15f2a5c083d0"  # Replace with your OpenWeatherMap API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        main = data['main']
        weather_desc = data['weather'][0]['description']
        temp = main['temp']
        humidity = main['humidity']
        
        return f"Current weather in {city}: {weather_desc}, temperature: {temp}°C, humidity: {humidity}%."
    elif response.status_code == 401:
        return "API key is not valid, please check your API key again."
    else:
        return "Ensure the city name is correct and matches the expected format for the weather API you're using."

# Function to get the current date and time
def get_current_date_time():
    now = datetime.now()
    return f"It is currently {now.strftime('%A, %B %d, %Y %H:%M:%S')}."

# Function to get nutrition information from Nutritionix API
def get_nutrition_info(food_item):
    api_key = "029884c961fd2cc90f58e5a8199141d7"  # Replace with your API key
    app_id = "796700de"  # Replace with your Nutritionix App ID
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"

    headers = {
        "x-app-id": app_id,
        "x-app-key": api_key,
        "Content-Type": "application/json"
    }

    data = {
        "query": food_item,
        "timezone": "US/Eastern"
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        nutrition_data = response.json()
        food = nutrition_data['foods'][0]
        name = food['food_name']
        calories = food['nf_calories']
        protein = food['nf_protein']
        fat = food['nf_total_fat']
        carbs = food['nf_total_carbohydrate']
        
        return f"Nutrition information for {name}: Calories: {calories} kcal, Protein: {protein}g, Fat: {fat}g, Carbohydrates: {carbs}g."
    else:
        return "Make sure the food name is spelled correctly. Ensure your API key is correct and working."

# Function to handle input related to weather, time, nutrition, or general queries
def chatbot_response(prompt, chatbot):
    if "weather" in prompt.lower():
        if "in" in prompt.lower():
            city = prompt.split("in")[-1].strip()
            return get_weather(city)
        else:
            return "Please mention the city to get weather information. For example, 'What is the weather in New York?'."
    elif "date" in prompt.lower() or "time" in prompt.lower():
        return get_current_date_time()
    elif "nutrition" in prompt.lower():
        # Get food from input after the keyword
        food_item = prompt.split("about")[-1].strip()
        return get_nutrition_info(food_item)
    else:
        return chatbot.chat_engine.chat(prompt).response

# Using cache for outer class function to avoid errors
@st.cache_resource(show_spinner=False)
def load_data(vector_store=None):
    with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes."):
        # Read & load document from folder
        reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
        documents = reader.load_data()

    if vector_store is None:
        index = VectorStoreIndex.from_documents(documents)
    return index

class Chatbot:
    def __init__(self, llm="llama3.1:latest", embedding_model="intfloat/multilingual-e5-large", vector_store=None):
        self.Settings = self.set_setting(llm, embedding_model)

        # Indexing
        self.index = load_data()

        # Memory
        self.memory = self.create_memory()

        # Chat Engine
        self.chat_engine = self.create_chat_engine(self.index)

    def set_setting(self, llm, embedding_model):
        Settings.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = FastEmbedEmbedding(
            model_name=embedding_model, cache_dir="./fastembed_cache")
        Settings.system_prompt = """
                                You are a multi-lingual expert system who has knowledge, based on 
                                real-time data. You will always try to be helpful and try to help them 
                                answering their question. If you don't know the answer, say that you DON'T
                                KNOW.
                                """

        return Settings

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        self.chat_store.store = {"chat_history": self.chat_history}

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

    def create_chat_engine(self, index):
        return CondensePlusContextChatEngine(
            verbose=True,
            memory=self.memory,
            retriever=index.as_retriever(),
            llm=Settings.llm
        )

# Main Program
st.title("Simple RAG Chatbot with Streamlit")
chatbot = Chatbot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello there 👋!\n\n Good to see you, how may I help you today? Feel free to ask me 😁"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chatbot.set_chat_history(st.session_state.messages)

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get chatbot response based on input
    with st.chat_message("assistant"):
        response = chatbot_response(prompt, chatbot)
        st.markdown(response)

    # Add assistant message to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
