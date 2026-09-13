import tomllib
from huggingface_hub import InferenceClient

with open(".streamlit/secrets.toml", "rb") as f:
    token = tomllib.load(f)["HF_TOKEN"]

client = InferenceClient(api_key=token, provider="novita")

response = client.chat_completion(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=50,
)

print(response.choices[0].message.content)