# from vllm import LLM, SamplingParams
import torch
def vllm_runner(args,messages):
    from openai import OpenAI
    # Set OpenAI's API key and API base to use vLLM's API server.
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8888/v1",

    )
    response = client.chat.completions.create(
        model="gemma-4-e4b-it",
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=[],
        n=args.n
    )
    responses = [response.choices[i].message.content for i in range(args.n)]
    return responses



if __name__ == '__main__':
    pass



