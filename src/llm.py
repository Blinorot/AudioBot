from pathlib import Path

from groq import Groq

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent

MAX_HISTORY_LENGTH = 100


def init_llm():
    api_path = ROOT_PATH / ".env"
    with api_path.open("r") as f:
        api_key = f.readline().strip()
    client = Groq(api_key=api_key)

    history = [
        {"role": "system", "content": "You are an intelligent ai voice assistant"}
    ]

    return client, history


def run_llm(text, history, client):
    history.append({"role": "user", "content": text})

    chat_completion = client.chat.completions.create(
        messages=history,
        model="llama3-8b-8192",
    )

    predicted_text = chat_completion.choices[0].message.content

    history.append({"role": "assistant", "content": predicted_text})

    return predicted_text, history
