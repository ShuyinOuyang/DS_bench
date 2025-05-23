from typing import Dict, Union
from aiohttp import ClientSession
from openai import OpenAI
from dataclasses import dataclass

class AsyncGPTError(Exception):
    pass
async def post(url, json, headers):
    async with ClientSession() as session:
        response = await session.post(url, json=json, headers=headers)
        response = await response.json()
    return response


@dataclass
class ChatCompletionChoice:
    index: int
    message: Dict[str, str]
    finish_reason: str

    def __str__(self) -> str:
        return self.message.content


@dataclass
class ChatCompletion:
    id: str
    created: int
    choices: list[ChatCompletionChoice]
    usage: Dict[str, int]
    object: str = "chat.completion"

    def __str__(self) -> str:
        return str(self.choices[0])

class AsyncGPT:
    def __init__(self, api_key: str, organization: str, model: str):
        self.api_key = api_key
        self.organization = organization
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.organization
        )
    @property
    def headers(self):
        if 'codestral' in self.model:
            return {"Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": "Bearer " + self.api_key}
        else:
            return {"Content-Type": "application/json",
                    "Authorization": "Bearer " + self.api_key}
    async def chat_complete(
            self, messages: list[Dict[str, str]],
            model: str = "gpt-3.5-turbo",
            temperature: float = 1.0, top_p: float = 1.0,
            stop: Union[str, list] = None, n: int = 1, stream: bool = False,
            max_tokens: int = None, presence_penalty: float = 0,
            frequency_penalty: float = 0, user: str = None):

        if not all((messages[0].get("role"), messages[0].get("content"))):
            raise ValueError("Invalid messages object")

        if 'codestral' in self.model:
            params = {
                        "model": model, "messages": messages,
                        "temperature": float(temperature),
                        "stream": bool(stream),
                        "max_tokens": 2000,
                    }
        else:
            params = {
                        "model": model, "messages": messages,
                        "temperature": float(temperature), "top_p": float(top_p),
                        "stop": stop, "n": int(n), "stream": bool(stream),
                        "max_tokens": max_tokens,
                        "presence_penalty": float(presence_penalty),
                        "frequency_penalty": float(frequency_penalty)
                    }
        if user:
            params["user"] = user

        if 'deepseek' in self.model:
            response = await post(
                "https://api.deepseek.com/chat/completions", json=params, headers=self.headers)
        elif 'codestral' in self.model:
            # print(params)
            # print(self.headers)
            # response = await post(
            #     "https://codestral.mistral.ai/v1/chat/completions", json=params, headers=self.headers)
            response = await post(
                "https://api.mistral.ai/v1/chat/completions", json=params, headers=self.headers)
        else:
            response = await post(
                "https://api.openai.com/v1/chat/completions", json=params, headers=self.headers)

        return response