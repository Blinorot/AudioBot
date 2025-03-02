from pathlib import Path

from groq import Groq

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent


def init_llm():
    """
    Initialize LLM client and history using Groq and a system prompt.

    You should save your API key in the .env file outside of the src dir.

    Returns:
        client: Groq client used for API calls.
        history: list of dictionaries, containing chat history.
            Initialized with a system prompt.
    """
    api_path = ROOT_PATH / ".env"
    with api_path.open("r") as f:
        api_key = f.readline().strip()
    client = Groq(api_key=api_key)

    history = [
        {
            "role": "system",
            "content": "You are an intelligent ai voice assistant named Sheila",
        }
    ]

    return client, history


def run_llm(text, history, client, config):
    """
    Get response from LLM for a user query.

    Args:
        text (str): input user query.
        history (list[dict]): chat history.
        client: Groq client for API calls.
        config: Hydra config with llm configuration in config.llm.
    Returns:
        predicted_text (str): LLM response.
        history (list[dict]): updated history.
    """
    max_tokens = config.llm.max_tokens
    max_history = config.llm.max_history

    if len(history) >= max_history + 1:
        history = [history[0]] + history[-(max_history - 1) :]

    history.append({"role": "user", "content": text})

    chat_completion = client.chat.completions.create(
        messages=history,
        model=config.llm.model_id,
        max_tokens=max_tokens,
    )

    predicted_text = chat_completion.choices[0].message.content

    history.append({"role": "assistant", "content": predicted_text})

    return predicted_text, history
