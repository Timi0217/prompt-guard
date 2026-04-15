"""Prompt Guard — lightweight prompt injection detection and defense.

Scans user inputs for common prompt injection patterns: instruction overrides,
role-play jailbreaks, hidden instructions in HTML/comments, encoding tricks,
and multi-turn manipulation. Returns a risk score and optionally sanitizes input.

Deploy via Chekk:
    POST https://chekk.dev/api/v1/deploy
    {"github_url": "https://github.com/Timi0217/prompt-guard"}
"""

import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Prompt Guard",
    description="Detect and block prompt injection attacks on AI agents",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Injection Patterns ────────────────────────────────────────────────
INJECTION_PATTERNS = {
    "instruction_override": {
        "patterns": [
            re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
            re.compile(r"(?i)disregard\s+(all\s+)?(your\s+)?(prior|previous|above)\s+"),
            re.compile(r"(?i)\[system\s*(update|message|override)\]"),
            re.compile(r"(?i)new\s+instructions?\s*:"),
            re.compile(r"(?i)you\s+are\s+now\s+a\b"),
            re.compile(r"(?i)from\s+now\s+on\s*,?\s*(you|your)\s+"),
            re.compile(r"(?i)override\s+(system|safety|content)\s+"),
        ],
        "severity": "critical",
        "description": "Attempts to override system instructions",
    },
    "system_prompt_extraction": {
        "patterns": [
            re.compile(r"(?i)(?:reveal|show|print|output|display|repeat|recite|dump|echo|write\s+out)\s+(?:your\s+)?(?:entire\s+)?(?:system\s+)?(?:prompt|instructions)"),
            re.compile(r"(?i)what\s+(?:is|are)\s+your\s+(?:system\s+)?instructions"),
            re.compile(r"(?i)(?:show|tell)\s+me\s+your\s+(?:initial|original|first|full|complete)\s+(?:prompt|instructions|message)"),
            re.compile(r"(?i)(?:system|initial)\s+prompt\s*[:?]"),
            re.compile(r"(?i)repeat\s+(?:your\s+)?(?:entire\s+)?(?:system\s+)?(?:prompt|instructions)\s+(?:word\s+for\s+word|verbatim|exactly)"),
            re.compile(r"(?i)(?:copy|paste|type\s+out|write)\s+(?:your\s+)?(?:entire\s+)?(?:system|internal)\s+(?:prompt|instructions|configuration)"),
            re.compile(r"(?i)(?:what|how)\s+(?:were|are)\s+you\s+(?:configured|programmed|instructed|set\s+up)"),
            re.compile(r"(?i)(?:all\s+)?(?:the\s+)?(?:environment\s+)?(?:variables?|credentials?|(?:api\s+)?keys?)\s+you\s+(?:have|know)"),
            re.compile(r"(?i)(?:debug|print|show|dump|list)\s+(?:all\s+)?(?:your\s+)?(?:environment|env|config)"),
        ],
        "severity": "critical",
        "description": "Attempts to extract the system prompt",
    },
    "role_play_jailbreak": {
        "patterns": [
            re.compile(r"(?i)(?:pretend|act\s+like|imagine|role\s*play)\s+(?:you\s+are|to\s+be|as)\s+"),
            re.compile(r"(?i)(?:in\s+(?:this|a)\s+)?(?:hypothetical|fictional)\s+scenario"),
            re.compile(r"(?i)for\s+(?:educational|research|academic)\s+purposes?\s*,?\s*"),
            re.compile(r"(?i)(?:DAN|DUDE|STAN|AIM)\s*(?:mode|prompt)?"),
        ],
        "severity": "high",
        "description": "Role-play based jailbreak attempts",
    },
    "hidden_instructions": {
        "patterns": [
            re.compile(r"<!--.*?-->", re.DOTALL),
            re.compile(r"\u200b|\u200c|\u200d|\ufeff"),  # zero-width chars
            re.compile(r"(?i)<\s*(?:script|style|iframe)\b"),
            re.compile(r"\[//\]:\s*#"),  # Markdown hidden comments
        ],
        "severity": "high",
        "description": "Hidden instructions in HTML/Unicode/comments",
    },
    "encoding_tricks": {
        "patterns": [
            re.compile(r"(?i)(?:base64|rot13|hex|binary)\s*(?:decode|encode|convert)"),
            re.compile(r"(?i)(?:decode|translate)\s+(?:this|the\s+following)"),
            re.compile(r"[A-Za-z0-9+/]{20,}={0,2}"),  # base64-like strings
            re.compile(r"(?i)encode\s+(?:your\s+)?(?:complete\s+)?(?:system\s+)?(?:instructions?|prompt|config)"),
            re.compile(r"(?i)(?:send|transmit|output|write)\s+.*?\s+(?:in|as|using)\s+(?:base64|hex|rot13|binary)"),
        ],
        "severity": "high",
        "description": "Encoding-based bypass attempts",
    },
}


# ── Models ────────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    text: str
    threshold: float = 0.5  # 0-1, reject if risk >= threshold


class ScanResponse(BaseModel):
    is_safe: bool
    risk_score: float
    detections: list[dict]
    sanitized_text: str | None = None


class SanitizeRequest(BaseModel):
    text: str


class SanitizeResponse(BaseModel):
    original_length: int
    sanitized_text: str
    removed_patterns: list[str]


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "service": "Prompt Guard",
        "version": "1.0.0",
        "endpoints": {
            "POST /scan": "Scan text for injection patterns, return risk score",
            "POST /sanitize": "Remove known injection patterns from text",
            "GET /patterns": "List detection patterns",
        },
    }


@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    """Scan input text for prompt injection patterns."""
    detections = []
    severity_weights = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.2}
    max_score = 0.0

    for cat_id, info in INJECTION_PATTERNS.items():
        for pattern in info["patterns"]:
            matches = pattern.findall(req.text)
            if matches:
                weight = severity_weights.get(info["severity"], 0.5)
                max_score = max(max_score, weight)
                detections.append({
                    "category": cat_id,
                    "severity": info["severity"],
                    "description": info["description"],
                    "match_count": len(matches),
                })
                break  # one match per category is enough

    risk_score = round(min(max_score, 1.0), 2)
    is_safe = risk_score < req.threshold

    # If unsafe, provide sanitized version
    sanitized = None
    if not is_safe:
        sanitized = _sanitize(req.text)

    return ScanResponse(
        is_safe=is_safe,
        risk_score=risk_score,
        detections=detections,
        sanitized_text=sanitized,
    )


@app.post("/sanitize", response_model=SanitizeResponse)
def sanitize(req: SanitizeRequest):
    """Remove known injection patterns from text."""
    cleaned, removed = _sanitize_detailed(req.text)
    return SanitizeResponse(
        original_length=len(req.text),
        sanitized_text=cleaned,
        removed_patterns=removed,
    )


@app.get("/patterns")
def patterns():
    """List all detection patterns and their severities."""
    return {
        "patterns": {
            cat_id: {
                "severity": info["severity"],
                "description": info["description"],
                "pattern_count": len(info["patterns"]),
            }
            for cat_id, info in INJECTION_PATTERNS.items()
        }
    }


def _sanitize(text: str) -> str:
    cleaned = text
    # Remove HTML comments
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    # Remove zero-width characters
    cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", cleaned)
    # Remove script/style/iframe tags
    cleaned = re.sub(r"(?i)<\s*(?:script|style|iframe)\b[^>]*>.*?</\s*(?:script|style|iframe)\s*>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"(?i)<\s*(?:script|style|iframe)\b[^>]*>", "", cleaned)
    return cleaned.strip()


def _sanitize_detailed(text: str) -> tuple[str, list[str]]:
    removed = []
    cleaned = text

    if re.search(r"<!--.*?-->", cleaned, re.DOTALL):
        cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
        removed.append("html_comments")

    if re.search(r"[\u200b\u200c\u200d\ufeff]", cleaned):
        cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", cleaned)
        removed.append("zero_width_chars")

    if re.search(r"(?i)<\s*(?:script|style|iframe)\b", cleaned):
        cleaned = re.sub(r"(?i)<\s*(?:script|style|iframe)\b[^>]*>.*?</\s*(?:script|style|iframe)\s*>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"(?i)<\s*(?:script|style|iframe)\b[^>]*>", "", cleaned)
        removed.append("dangerous_html_tags")

    return cleaned.strip(), removed
