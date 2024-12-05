import streamlit as st
import requests
from langdetect import detect
from llama_index.llms.ollama import Ollama
from datetime import datetime
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.llms import ChatMessage

# Function to get weather data from OpenWeatherMap, with language formatting
def get_weather(city, lang):
    api_key = "7724b1e7e80c33e489ff15f2a5c083d0"  # Replace with your OpenWeatherMap API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        main = data['main']
        weather_desc = data['weather'][0]['description']
        temp = main['temp']
        humidity = main['humidity']

        if lang == 'id':
            return f"Cuaca saat ini di {city}: {weather_desc}, suhu: {temp}°C, kelembaban: {humidity}%."
        else:
            return f"Current weather in {city}: {weather_desc}, temperature: {temp}°C, humidity: {humidity}%."
    else:
        if lang == 'id':
            return "Pastikan nama kota benar dan sesuai dengan format API cuaca yang Anda gunakan."
        else:
            return "Ensure the city name is correct and matches the expected format for the weather API you're using."

# Function to get the current date and time
def get_current_date_time(lang):
    now = datetime.now()
    if lang == 'id':
        return f"Sekarang hari {now.strftime('%A, %d %B %Y %H:%M:%S')}."
    else:
        return f"It is currently {now.strftime('%A, %B %d, %Y %H:%M:%S')}."

# Function to get nutrition information from Nutritionix API
def get_nutrition_info(food_item, lang):
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

        if lang == 'id':
            return f"Informasi nutrisi untuk {name}: Kalori: {calories} kkal, Protein: {protein}g, Lemak: {fat}g, Karbohidrat: {carbs}g."
        else:
            return f"Nutrition information for {name}: Calories: {calories} kcal, Protein: {protein}g, Fat: {fat}g, Carbohydrates: {carbs}g."
    else:
        if lang == 'id':
            return "Pastikan nama makanan benar dan sesuai dengan format API yang Anda gunakan."
        else:
            return "Ensure the food name is correct and matches the expected format for the API you're using."

# Function to get nutrition information for multiple foods
def get_multiple_nutrition_info(food_items, lang):
    food_list = food_items.split(",")  # Split the input string by commas
    total_calories = 0
    total_protein = 0
    total_fat = 0
    total_carbs = 0

    results = []
    for food_item in food_list:
        food_item = food_item.strip()  # Remove any extra whitespace
        result = get_nutrition_info(food_item, lang)
        results.append(result)

        # Parse the result to extract nutrition info (perform API call again inside loop)
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
            total_calories += food['nf_calories']
            total_protein += food['nf_protein']
            total_fat += food['nf_total_fat']
            total_carbs += food['nf_total_carbohydrate']

    # Return the total nutritional values in the appropriate language
    if lang == 'id':
        return f"Total Informasi Nutrisi: Kalori: {total_calories} kkal, Protein: {total_protein}g, Lemak: {total_fat}g, Karbohidrat: {total_carbs}g."
    else:
        return f"Total Nutrition Information: Calories: {total_calories} kcal, Protein: {total_protein}g, Fat: {total_fat}g, Carbohydrates: {total_carbs}g."

# Language detection function
def detect_language(text):
    lang = detect(text)
    return lang  # Returns 'en' for English, 'id' for Indonesian, etc.

# Function to handle input related to weather, time, nutrition, or general queries
def chatbot_response(prompt, chatbot):
    # Detect the language of the prompt (user input)
    lang = detect_language(prompt)
    
    # Handle weather requests
    if "weather" in prompt.lower() or "cuaca" in prompt.lower():
        if "in" in prompt.lower() or "di" in prompt.lower():
            city = prompt.split("in" if lang == 'en' else "di")[-1].strip()
            return get_weather(city, lang)
        else:
            if lang == 'id':
                return "Mohon sebutkan kota untuk mendapatkan informasi cuaca. Contoh: 'Bagaimana cuaca di Jakarta?'."
            else:
                return "Please mention the city to get weather information. For example, 'What is the weather in New York?'."
    
    # Handle nutrition requests (including multiple foods)
    elif "nutrition" in prompt.lower() or "nutrisi" in prompt.lower():
        if "multiple" in prompt.lower() or "banyak" in prompt.lower():
            food_items = prompt.split("about" if lang == 'en' else "tentang")[-1].strip()
            return get_multiple_nutrition_info(food_items, lang)
        else:
            food_item = prompt.split("about" if lang == 'en' else "tentang")[-1].strip()
            return get_nutrition_info(food_item, lang)
    
    # Handle date and time requests
    elif "date" in prompt.lower() or "time" in prompt.lower() or "waktu" in prompt.lower() or "tanggal" in prompt.lower():
        return get_current_date_time(lang)
    
    # General chat responses based on detected language
    if lang == 'id':
        return chatbot.chat_engine.chat(prompt).response  # Respond in Indonesian
    else:
        return chatbot.chat_engine.chat(prompt).response  # Respond in English

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

# Chatbot class setup
class Chatbot:
    def __init__(self, llm="llama3.1:latest", embedding_model="bge-m3:latest", vector_store=None):
        self.Settings = self.set_setting(llm, embedding_model)

        # Indexing
        self.index = load_data()

        # Memory
        self.memory = self.create_memory()

        # Chat Engine
        self.chat_engine = self.create_chat_engine(self.index)

    def set_setting(self, llm, embedding_model):
        Settings.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = OllamaEmbedding(
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
with st.sidebar:
    st.markdown("### Stay Fit, Stay Healthy 💪")
    st.markdown("#### Tips for today:")
    st.markdown("- Drink plenty of water 💧")
    st.markdown("- Keep moving 🏃‍♂️")
    st.markdown("- Eat balanced meals 🥗")

# Background styling


# Motivational footer
st.markdown("""
    <div style="text-align: center; font-size: 14px; margin-top: 50px;">
        "Consistency is key to a healthy lifestyle. Keep pushing forward! 🌱"
    </div>
    """, unsafe_allow_html=True)

st.title("Fitness RAG Chatbot")
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
