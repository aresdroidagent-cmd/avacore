from __future__ import annotations

import re
from dataclasses import dataclass

from avacore.core.jspace import JSpaceItem
from avacore.core.research_queue import ResearchTopic, stable_research_id

IGNORED_SOURCES = {"identity", "operating_rule"}
IGNORED_KINDS = {"self_anchor", "memory_rule", "identity", "operating_rule"}
GREETING_PATTERN = re.compile(
    r"^(hallo|hi|hey|guten (morgen|tag|abend)|hello|thanks|danke)[!., ]*$",
    re.IGNORECASE,
)
INTERNAL_PROMPT_MARKERS = {
    "übersetze die folgende kamerabeschreibung",
    "formuliere sie kurz, sachlich und natürlich",
    "erfinde keine zusätzlichen details",
    "beschreibung:",
    "you are looking at a live indoor camera image",
}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class CuriosityConfig:
    curiosity_weight: float = 0.15


def is_researchable_item(item: JSpaceItem) -> bool:
    content = " ".join((item.content or "").split())
    lowered = content.lower()
    if item.source in IGNORED_SOURCES or item.kind in IGNORED_KINDS:
        return False
    if len(content) < 24 or len(content.split()) < 4:
        return False
    if GREETING_PATTERN.match(content):
        return False
    if any(marker in lowered for marker in INTERNAL_PROMPT_MARKERS):
        return False
    return item.source != "research" and item.kind != "finding"


def normalize_to_question(content: str) -> str:
    content = " ".join((content or "").strip().split())
    content = re.sub(r"^(ava[,:\s]+)", "", content, flags=re.IGNORECASE)
    if not content:
        return ""
    if "?" in content:
        return content[: content.index("?") + 1]
    trimmed = content.rstrip(".! ")
    return (
        "Welche aktuellen, verlässlichen Informationen sind für dieses Thema "
        f"relevant: {trimmed}?"
    )


def short_title(question: str) -> str:
    cleaned = re.sub(
        r"^Welche aktuellen, verlässlichen Informationen sind für dieses Thema relevant:\s*",
        "",
        question,
        flags=re.IGNORECASE,
    ).rstrip("? ")
    return (cleaned or question).strip()[:100]


def score_item(
    item: JSpaceItem,
    curiosity_weight: float = 0.15,
) -> tuple[float, dict[str, float]]:
    text = item.content.lower()
    tags = {tag.lower() for tag in item.tags}
    activation = clamp(item.activation)
    priority = clamp(item.priority)

    question_signal = 1.0 if "?" in item.content else 0.65
    project_signal = 1.0 if tags.intersection(
        {"avacore", "ava", "robotics", "industrial", "vision", "programming", "devops"}
    ) else 0.55
    knowledge_gap = clamp(question_signal * 0.65 + project_signal * 0.35)

    freshness_terms = {
        "aktuell", "neueste", "heute", "version", "update", "news",
        "current", "latest", "recent", "release", "security",
    }
    freshness_need = (
        0.9
        if any(re.search(rf"\b{re.escape(term)}\b", text) for term in freshness_terms)
        else 0.35
    )

    urgency_terms = {
        "dringend", "blockiert", "fehler", "ausfall", "sicherheit",
        "urgent", "blocked", "error", "failure", "security",
    }
    urgency = (
        0.9
        if any(re.search(rf"\b{re.escape(term)}\b", text) for term in urgency_terms)
        else clamp(priority * 0.45)
    )

    curiosity = clamp(0.35 + min(len(tags), 6) * 0.07 + (0.15 if "?" in item.content else 0.0))
    base = (
        activation * 0.30
        + priority * 0.25
        + knowledge_gap * 0.20
        + freshness_need * 0.15
        + urgency * 0.10
    )
    bounded_weight = clamp(curiosity_weight) * 0.25
    score = clamp(base + curiosity * bounded_weight * (1.0 - base))
    components = {
        "activation": activation,
        "priority": priority,
        "knowledge_gap": knowledge_gap,
        "freshness_need": freshness_need,
        "urgency": urgency,
        "curiosity": curiosity,
    }
    return score, components


def topic_from_jspace_item(
    item: JSpaceItem,
    curiosity_weight: float = 0.15,
) -> ResearchTopic | None:
    if not is_researchable_item(item):
        return None
    question = normalize_to_question(item.content)
    if not question:
        return None
    score, components = score_item(item, curiosity_weight=curiosity_weight)
    return ResearchTopic(
        id=stable_research_id(question),
        title=short_title(question),
        question=question,
        origin="jspace",
        origin_item_ids=[item.id],
        tags=sorted(set(item.tags)),
        score=score,
        score_components=components,
    )


def derive_topics(
    items: list[JSpaceItem],
    curiosity_weight: float = 0.15,
) -> list[ResearchTopic]:
    topics: list[ResearchTopic] = []
    seen: set[str] = set()
    for item in items:
        topic = topic_from_jspace_item(item, curiosity_weight=curiosity_weight)
        if topic and topic.id not in seen:
            topics.append(topic)
            seen.add(topic.id)
    return sorted(topics, key=lambda topic: (-topic.score, topic.id))
