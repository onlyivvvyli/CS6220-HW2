
# --- Model callers ---
def call_openai(model_id: str, prompt: str, temperature=0.7, top_p=1, max_tokens=4096) -> str:
    from openai import OpenAI

    client = OpenAI(api_key="MY_API_KEY")
    resp = client.chat.completions.create(
        model= model_id,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def call_hf_inference(model_id: str, prompt: str, temperature=0.7, top_p=1, max_tokens=2048) -> str:
    from huggingface_hub import InferenceClient

    hf_token = "MY_HF_TOKEN"
    client = InferenceClient(
        provider="together",
        api_key=hf_token,
    )

    completion = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    return completion.choices[0].message["content"]

def call_gpt4o(prompt, **kw):
    return call_openai("gpt-4o", prompt, **kw)

def call_gpt4o_mini(prompt, **kw):
    return call_openai("gpt-4o-mini", prompt, **kw)

def call_gpt4(prompt, **kw):
    return call_openai("gpt-4.1", prompt, **kw)

def call_gpt_turbo(prompt, **kw):
    return call_openai("gpt-3.5-turbo", prompt, **kw)

def call_qwen25_7b(prompt, **kw):
    return call_hf_inference("Qwen/Qwen2.5-7B-Instruct", prompt, **kw)


if __name__ == "__main__":
    prompt = '''MY_PROMPT'''

    gpt4o = call_gpt4o(prompt)
    print(gpt4o)

    mini = call_gpt4o_mini(prompt)
    print(mini)

    qwen = call_qwen25_7b(prompt)
    print(qwen)

    gpt4 = call_gpt4(prompt)
    print(gpt4)

    turbo = call_gpt_turbo(prompt)
    print(turbo)