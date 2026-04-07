from pydantic import BaseModel


class IncomingMessage(BaseModel):
    channel: str
    user_id: str
    chat_id: str
    text: str
    timestamp: int


class AssistantReply(BaseModel):
    reply_text: str


class HealthStatus(BaseModel):
    ok: bool
    model: str
    profile: str
    max_history_turns: int
    ollama_url: str
