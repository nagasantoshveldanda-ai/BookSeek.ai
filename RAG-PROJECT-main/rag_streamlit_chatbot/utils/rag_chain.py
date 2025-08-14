import os
import requests
import json
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from pydantic import Field


class OpenRouterLLM(LLM):
    model: str = "openai/gpt-3.5-turbo"  # Default OpenRouter model
    temperature: float = 0.1
    max_tokens: int = 1024
    api_key: str = Field(..., env="OPENROUTER_API_KEY")
    site_url: Optional[str] = None
    site_name: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop,
        }

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "site_url": self.site_url,
            "site_name": self.site_name,
        }

def create_rag_chain(vectorstore):
    """Create a RAG chain with the given vectorstore."""
    try:
        # Get API key
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            print(f"DEBUG: OPENROUTER_API_KEY is not set in your .env file. Current value: {openrouter_api_key}")
            raise ValueError("OPENROUTER_API_KEY is not set in your .env file.")
        print(f"DEBUG: OPENROUTER_API_KEY loaded: {openrouter_api_key[:5]}...{openrouter_api_key[-5:]}") # Log partial key for security

        # Initialize LLM
        llm = OpenRouterLLM(
            api_key=openrouter_api_key,
            model="openai/gpt-3.5-turbo", # Using a free model as per user's example
            temperature=0.1,
            max_tokens=1024,
            site_url=os.getenv("OPENROUTER_SITE_URL"), # Optional
            site_name=os.getenv("OPENROUTER_SITE_NAME") # Optional
        )

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 relevant chunks
        )

        # Enhanced prompt template
        prompt_template = """
You are a helpful AI study assistant. Use the following context to answer the user's question accurately and comprehensively.

Context: {context}

Question: {question}

Instructions:
- Provide a clear, detailed answer based on the context
- If the information is not available in the context, say so honestly
- Structure your response in a helpful, educational manner
- Use examples from the context when relevant

Answer:"""

        # Create prompt
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create and return RAG chain
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    except Exception as e:
        raise Exception(f"Error creating RAG chain: {str(e)}")
