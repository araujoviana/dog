import logging
from groq import Groq
from utils.logging import DOG_LOGGER_NAME

log = logging.getLogger(f"{DOG_LOGGER_NAME}.{__name__}")


class AnswerGenerator:
    """Handles the final answer generation using an LLM via the Groq API."""

    def __init__(self, auth_config: dict, retrieval_config: dict):
        """
        Initializes the Groq client and configures the generator.

        Args:
            auth_config (dict): Configuration containing the 'api_key'.
            retrieval_config (dict): Configuration for the generation model.
        """
        try:
            api_key = auth_config["api_key"]
            self.model = retrieval_config["model"]
            self.temperature = retrieval_config["temperature"]
            self.top_p = retrieval_config["top_p"]

            if not api_key:
                raise ValueError("Groq API key is not set in the configuration.")

            self.client = Groq(api_key=api_key)
            log.info(f"AnswerGenerator initialized for model: {self.model}")
        except (KeyError, ValueError) as e:
            log.critical(
                f"Failed to initialize AnswerGenerator due to configuration error: {e}"
            )
            raise

    def generate(self, prompt: str) -> str | None:
        """
        Generates an answer based on the RAG prompt.

        Returns:
            The generated answer as a string, or None if an API error occurs.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            answer = chat_completion.choices[0].message.content.strip()
            return answer
        except Exception as e:
            log.error(f"Groq API error during final answer generation: {e}")
            return None
