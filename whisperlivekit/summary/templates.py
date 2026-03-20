"""Summary template management."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Template storage directory
TEMPLATES_DIR = Path.home() / ".wlk" / "templates"


@dataclass
class SummaryTemplate:
    """A summary template with system and user prompts."""

    id: str
    name: str
    description: str = ""
    system_prompt: str = ""
    user_prompt: str = "{{text}}"  # Template with placeholders
    slots: List[str] = field(default_factory=lambda: ["text"])

    def __post_init__(self):
        """Extract slots from user_prompt if not provided."""
        if not self.slots:
            self.slots = self._extract_slots(self.user_prompt)

    @staticmethod
    def _extract_slots(template: str) -> List[str]:
        """Extract {{slot}} placeholders from template."""
        pattern = r"\{\{(\w+)\}\}"
        return list(set(re.findall(pattern, template)))

    def render_user_prompt(self, **kwargs) -> str:
        """Render user prompt with provided values."""
        result = self.user_prompt
        for key, value in kwargs.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def to_dict(self) -> dict:
        """Serialize template to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "slots": self.slots,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SummaryTemplate":
        """Deserialize template from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt", ""),
            user_prompt=data.get("user_prompt", "{{text}}"),
            slots=data.get("slots", []),
        )


# Built-in templates
BUILTIN_TEMPLATES: Dict[str, SummaryTemplate] = {
    "meeting_minutes": SummaryTemplate(
        id="meeting_minutes",
        name="会议纪要",
        description="适用于会议记录，提取关键决策和行动项",
        system_prompt=(
            "你是一个会议助手。请用简洁的语言总结以下会议内容，"
            "提取关键决策、行动项和重要信息。"
            "总结应该简短（1-2句话），便于快速浏览。"
        ),
        user_prompt="{{text}}",
    ),
    "interview": SummaryTemplate(
        id="interview",
        name="面试留档",
        description="适用于面试记录，提取候选人回答要点",
        system_prompt=(
            "你是一个面试记录助手。请总结以下面试对话，"
            "提取候选人的关键回答要点和技能亮点。"
            "总结应该简短（1-2句话），便于快速回顾。"
        ),
        user_prompt="{{text}}",
    ),
    "general": SummaryTemplate(
        id="general",
        name="通用总结",
        description="通用总结模板",
        system_prompt=("请用简洁的语言总结以下内容，提取关键信息。总结应该简短（1-2句话）。"),
        user_prompt="{{text}}",
    ),
}


def get_template(template_id: str) -> Optional[SummaryTemplate]:
    """Get a template by ID.

    First checks built-in templates, then custom templates from disk.

    Args:
        template_id: Template identifier

    Returns:
        SummaryTemplate if found, None otherwise
    """
    # Check built-in first
    if template_id in BUILTIN_TEMPLATES:
        return BUILTIN_TEMPLATES[template_id]

    # Check custom templates
    template_file = TEMPLATES_DIR / f"{template_id}.json"
    if template_file.exists():
        try:
            import json

            with open(template_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return SummaryTemplate.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load template {template_id}: {e}")

    return None


def list_templates() -> List[SummaryTemplate]:
    """List all available templates (built-in + custom).

    Returns:
        List of SummaryTemplate objects
    """
    templates = list(BUILTIN_TEMPLATES.values())

    # Add custom templates
    if TEMPLATES_DIR.exists():
        for template_file in TEMPLATES_DIR.glob("*.json"):
            try:
                import json

                with open(template_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                templates.append(SummaryTemplate.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load template from {template_file}: {e}")

    return templates


def save_template(template: SummaryTemplate) -> bool:
    """Save a custom template to disk.

    Args:
        template: Template to save

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        template_file = TEMPLATES_DIR / f"{template.id}.json"

        import json

        with open(template_file, "w", encoding="utf-8") as f:
            json.dump(template.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Saved template: {template.id}")
        return True

    except Exception as e:
        logger.error(f"Failed to save template {template.id}: {e}")
        return False


def delete_template(template_id: str) -> bool:
    """Delete a custom template from disk.

    Note: Cannot delete built-in templates.

    Args:
        template_id: Template identifier

    Returns:
        True if deleted successfully, False otherwise
    """
    if template_id in BUILTIN_TEMPLATES:
        logger.warning(f"Cannot delete built-in template: {template_id}")
        return False

    template_file = TEMPLATES_DIR / f"{template_id}.json"
    if template_file.exists():
        try:
            template_file.unlink()
            logger.info(f"Deleted template: {template_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            return False

    return False
