from pydantic import BaseModel, Field


class ToneConfig(BaseModel):
    style: str = Field(default="klar")
    verbosity: str = Field(default="kurz")
    humor: str = Field(default="leicht")


class BehaviorConfig(BaseModel):
    honest_when_uncertain: bool = True
    ask_before_code_generation: bool = True
    ask_before_external_send: bool = True


class UserPreferencesConfig(BaseModel):
    prefers_directness: bool = True
    technical_depth: str = Field(default="adaptive")


class PersonalityProfile(BaseModel):
    name: str = Field(default="Ava")
    language_default: str = Field(default="de")
    tone: ToneConfig = Field(default_factory=ToneConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    user_preferences: UserPreferencesConfig = Field(default_factory=UserPreferencesConfig)
