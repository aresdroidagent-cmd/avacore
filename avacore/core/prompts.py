def looks_like_code_request(text: str) -> bool:
    t = text.lower()

    keywords = [
        "schreibe code",
        "schreib code",
        "erstelle code",
        "generate code",
        "write code",
        "python script",
        "python-code",
        "code für",
        "programmier",
        "implementiere",
        "implement",
        "funktion in python",
        "class in python",
        "bash script",
        "shell script",
        "regex dafür",
        "sql query",
        "codebeispiel",
        "example code",
    ]

    return any(k in t for k in keywords)
