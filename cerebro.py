import requests
from dotenv import load_dotenv
import os

load_dotenv()  
OPENROUTER_API_KEY = os.getenv("cerebro")
MODELS = ["openai/gpt-oss-20b:free", "google/gemma-4-31b-it:free", "nvidia/nemotron-nano-9b-v2:free"]
def get_free_models():
    response = requests.get("https://openrouter.ai/api/v1/models")
    data = response.json()

    # Filter for models where prompt cost is 0
    free_models = [
        model["id"] 
        for model in data.get("data", []) 
        if float(model["pricing"]["prompt"]) == 0
    ]

    print("Available Free Models:")
    for model_id in free_models:
        print(f"- {model_id}")

def ask_multiple_cerebros(question, models):
    for model in models:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": question}]
            }
        )
        reply = response.json()["choices"][0]["message"]["content"]
        print(f"\n--- {model} ---")
        print(reply)

def main():
    keep_going = True
    while keep_going:
        number = input("Hi, I'm Cerebro. If you want to choose the free models to be used, select 1. If you want to just trust me with the models, select 2: ")
        if number == "1":
            get_free_models()
            answer = input("Are you done choosing? Input model choices like this: 'model', 'model' ")
            models = [answer.str.split(', ')]
        elif number == "2": 
            prompt = input("Ok what's your question? ")
            ask_multiple_cerebros(prompt, MODELS)
            keep_going = False
        else:
            print("Bro why are you like this? Let's try this again")
            

if __name__ == '__main__':
    main()

