from main import OllamaDefinitionProvider
import pytest


def test_ollama_provider():
    provider = OllamaDefinitionProvider()
    definitions=provider.get_definitions(["manger"],15)
    pass
