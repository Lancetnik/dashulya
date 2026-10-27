"""Provider policy for the memory embedding model."""

from openai import AsyncOpenAI, OpenAIError


class MemoryEmbeddings:
    def __init__(
        self, client: AsyncOpenAI, *, model: str = "text-embedding-3-small"
    ) -> None:
        self.client = client
        self.model = model

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model,
                dimensions=512,
                encoding_format="float",
            )
        except OpenAIError:
            error_message = "Memory embedding request failed."
            raise RuntimeError(error_message) from None
        if sorted(item.index for item in response.data) != list(range(len(texts))):
            error_message = "Memory embedding response is incomplete."
            raise ValueError(error_message)
        return [
            item.embedding for item in sorted(response.data, key=lambda item: item.index)
        ]
