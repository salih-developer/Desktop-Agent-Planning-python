from __future__ import annotations

import ollama


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self._model = model
        self._client = ollama.Client(host=base_url)

    def embed(self, text: str) -> list[float]:
        response = self._client.embed(model=self._model, input=text)
        if not response.embeddings:
            raise RuntimeError(
                f"Embedding modeli '{self._model}' boş döndü. "
                f"Ollama'da model yüklü mü? (ollama pull {self._model})"
            )
        return response.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(model=self._model, input=texts)
        return response.embeddings
