from pydantic import BaseModel, Field
from typing import Literal


PolicyMode = Literal["allow", "deny", "ask", "log_only"]
PolicyScope = Literal["global", "channel", "user"]


class PolicyRule(BaseModel):
    domain: str = Field(...)
    action: str = Field(...)
    mode: PolicyMode = Field(...)
    scope_type: PolicyScope = Field(default="global")
    scope_value: str = Field(default="*")
    reason: str = Field(default="")
