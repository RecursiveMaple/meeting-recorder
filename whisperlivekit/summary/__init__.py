"""Summary module for meeting recorder."""

from .llm_client import LLMClient
from .templates import SummaryTemplate, get_template, list_templates

__all__ = ["LLMClient", "SummaryTemplate", "get_template", "list_templates"]
