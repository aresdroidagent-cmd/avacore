LOW_VRAM_PROFILE = {
    "name": "low_vram",
    "default_model": "llama3.2:3b-32k",
    "fallback_model": "qwen2.5:3b-instruct-32k",
    "max_history_turns": 8,
    "request_timeout_ms": 120000,
    "tool_mode": "minimal",
}

MID_VRAM_PROFILE = {
    "name": "mid_vram",
    "default_model": "qwen2.5:7b-instruct",
    "fallback_model": "llama3.2:3b-32k",
    "max_history_turns": 12,
    "request_timeout_ms": 180000,
    "tool_mode": "standard",
}

CPU_FALLBACK_PROFILE = {
    "name": "cpu_fallback",
    "default_model": "llama3.2:3b",
    "fallback_model": "qwen2.5:3b-instruct",
    "max_history_turns": 4,
    "request_timeout_ms": 240000,
    "tool_mode": "minimal",
}

PROFILES = {
    "low_vram": LOW_VRAM_PROFILE,
    "mid_vram": MID_VRAM_PROFILE,
    "cpu_fallback": CPU_FALLBACK_PROFILE,
}
