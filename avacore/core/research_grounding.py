"""Deterministic temporal and question grounding for user-triggered research."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from avacore.tools.web_research import ResearchSource


class ResearchMode(str, Enum):
    FACTUAL = "FACTUAL"
    CURRENT_STATE = "CURRENT_STATE"
    COMPARISON = "COMPARISON"
    CURRENT_COMPARISON = "CURRENT_COMPARISON"


class TemporalScope(str, Enum):
    TIMELESS = "TIMELESS"
    CURRENT = "CURRENT"
    RECENT = "RECENT"
    DATE_SPECIFIC = "DATE_SPECIFIC"


class TemporalStatus(str, Enum):
    CONFIRMED_CURRENT = "CONFIRMED_CURRENT"
    TEMPORALLY_COMPATIBLE = "TEMPORALLY_COMPATIBLE"
    TEMPORAL_UNKNOWN = "TEMPORAL_UNKNOWN"
    TEMPORAL_CONFLICT = "TEMPORAL_CONFLICT"
    STALE = "STALE"


class TemporalBasis(str, Enum):
    EXPLICIT_DATE = "EXPLICIT_DATE"
    STRUCTURED_METADATA = "STRUCTURED_METADATA"
    VALIDITY_INTERVAL = "VALIDITY_INTERVAL"
    CURRENT_PAGE_MARKER = "CURRENT_PAGE_MARKER"
    TIMELESS_FACT = "TIMELESS_FACT"
    LIVE_AUTHORITATIVE_DOCUMENTATION = "LIVE_AUTHORITATIVE_DOCUMENTATION"
    UNKNOWN = "UNKNOWN"


class EvidenceVolatility(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RecommendationConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class EvidenceRole(str, Enum):
    PRIMARY = "PRIMARY"
    SUPPORTING = "SUPPORTING"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class TemporalContext:
    now_utc: datetime
    local_timezone: str
    local_date: date
    target_date: date | None

    @classmethod
    def create(cls, timezone_name: str, target_date: date | None = None,
               now_utc: datetime | None = None) -> "TemporalContext":
        now = now_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        local = now.astimezone(ZoneInfo(timezone_name)).date()
        return cls(now, timezone_name, local, target_date)


@dataclass(frozen=True)
class ResearchQuestionSpec:
    original_query: str
    research_mode: ResearchMode
    temporal_scope: TemporalScope
    target_date: date | None
    requires_fresh_evidence: bool
    answer_goal: str
    comparison_targets: tuple[str, ...] = ()
    location_text: str | None = None
    requested_dimensions: tuple[str, ...] = ()
    comparison_criteria: tuple[ComparisonCriterion, ...] = ()
    evidence_requirements: tuple[EvidenceRequirement, ...] = ()


@dataclass(frozen=True)
class ComparisonCriterion:
    criterion_id: str
    label: str
    source: str
    importance: float
    requires_both_targets: bool
    requires_current_evidence: bool


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    criterion: str
    target: str
    query_terms: tuple[str, ...]
    temporal_requirement: str
    evidence_type: str
    required: bool = True
    satisfied: bool = False
    supporting_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchSearchQuery:
    query_id: str
    query_text: str
    requirement_ids: tuple[str, ...] = ()
    target: str | None = None
    criterion: str | None = None


_DATE_ISO = re.compile(r"\b(20\d{2})-(0?[1-9]|1[0-2])-([0-2]?\d|3[01])\b")
_DATE_DMY = re.compile(r"\b([0-2]?\d|3[01])\.\s*(0?[1-9]|1[0-2])\.\s*(20\d{2})\b")
_MONTHS = {"january":1, "february":2, "march":3, "april":4, "may":5, "june":6,
           "july":7, "august":8, "september":9, "october":10, "november":11, "december":12,
           "januar":1, "februar":2, "märz":3, "april":4, "mai":5, "juni":6,
           "juli":7, "august":8, "september":9, "oktober":10, "november":11, "dezember":12}
_DATE_WORD = re.compile(r"\b(?:(\d{1,2})\.?\s+([a-zä]+)|([a-zä]+)\s+(\d{1,2}),?)\s+(20\d{2})\b", re.I)
_VERSION = re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]*)\s+(\d{1,2}(?:\.\d{1,2}){0,2})\b")
_GENERIC = {"was", "für", "eine", "oder", "welche", "würdest", "empfehlen", "moment", "heute",
            "aktuell", "local", "lokale", "llms", "llm", "applikationen", "application", "applications",
            "what", "which", "should", "recommend", "current", "today", "latest", "version"}
_QUERY_FILLER = _GENERIC | {"du", "im", "den", "die", "das", "der", "und", "mit", "von", "zum",
                            "bei", "mir", "ich", "in", "and", "for", "the", "a", "an", "vs"}


def _comparison_context_terms(query: str, targets: tuple[str, ...]) -> tuple[str, ...]:
    context = query
    for target in targets:
        context = re.sub(re.escape(target), " ", context, flags=re.I)
    words = [word for word in re.findall(r"[\wäöüß+-]{2,}", context.casefold())
             if word not in (_QUERY_FILLER - {"llm", "llms", "applikationen", "application", "applications"})
             and not word.isdigit()]
    return tuple(dict.fromkeys(words))[:8]


def research_search_queries(spec: ResearchQuestionSpec) -> tuple[ResearchSearchQuery, ...]:
    """Build at most four semantic queries from explicit evidence requirements."""
    if len(spec.comparison_targets) != 2:
        return (ResearchSearchQuery("q1", spec.original_query),)
    if not spec.evidence_requirements:
        context = " ".join(_comparison_context_terms(spec.original_query, spec.comparison_targets))
        a, b = spec.comparison_targets
        texts = (spec.original_query, f"{a} {context}".strip(), f"{b} {context}".strip(),
                 f"{a} {b} {context}".strip())
        return tuple(ResearchSearchQuery(f"q{index}", text) for index, text in enumerate(
            dict.fromkeys(texts), 1))[:4]
    queries = []
    for target in spec.comparison_targets:
        lifecycle = [r for r in spec.evidence_requirements
                     if r.target == target and r.criterion == "support_lifecycle"]
        if lifecycle:
            requirement = lifecycle[0]
            queries.append(ResearchSearchQuery(f"q{len(queries)+1}",
                           f"{target} {' '.join(requirement.query_terms)}", (requirement.requirement_id,),
                           target, requirement.criterion))
    for target in spec.comparison_targets:
        contextual = [r for r in spec.evidence_requirements
                      if r.target == target and r.criterion != "support_lifecycle"]
        terms = tuple(dict.fromkeys(term for requirement in contextual for term in requirement.query_terms))
        queries.append(ResearchSearchQuery(f"q{len(queries)+1}", f"{target} {' '.join(terms)}",
                                           tuple(r.requirement_id for r in contextual), target,
                                           "requested_context_compatibility+current_limitations"))
    return tuple(queries)


def comparison_search_queries(spec: ResearchQuestionSpec) -> tuple[str, ...]:
    """Compatibility wrapper returning bounded query text."""
    return tuple(query.query_text for query in research_search_queries(spec))


def extract_dates(text: str) -> set[date]:
    found: set[date] = set()
    for match in _DATE_ISO.finditer(text):
        try: found.add(date(int(match[1]), int(match[2]), int(match[3])))
        except ValueError: pass
    for match in _DATE_DMY.finditer(text):
        try: found.add(date(int(match[3]), int(match[2]), int(match[1])))
        except ValueError: pass
    for match in _DATE_WORD.finditer(text):
        day = match[1] or match[4]
        month = _MONTHS.get((match[2] or match[3]).casefold())
        if month:
            try: found.add(date(int(match[5]), month, int(day)))
            except ValueError: pass
    return found


def _metadata_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        dates = extract_dates(value)
        return next(iter(dates)) if len(dates) == 1 else None


def classify_research_question(query: str, temporal: TemporalContext) -> ResearchQuestionSpec:
    q = query.casefold()
    explicit = extract_dates(query)
    target_date = next(iter(explicit)) if len(explicit) == 1 else None
    today = bool(re.search(r"\b(heute|today)\b", q))
    current = bool(re.search(r"\b(aktuell|momentan|derzeit|im moment|neueste|latest|current|right now)\b", q))
    recent = bool(re.search(r"\b(kürzlich|recent|recently|letzte[nr]? woche)\b", q))
    scope = (TemporalScope.DATE_SPECIFIC if target_date else TemporalScope.CURRENT if today or current else
             TemporalScope.RECENT if recent else TemporalScope.TIMELESS)
    if today:
        target_date = temporal.local_date
    versions = [f"{m[1]} {m[2]}" for m in _VERSION.finditer(query)
                if m[1].casefold() not in {"vs", "versus", "oder", "or", "for", "für"}]
    targets = tuple(dict.fromkeys(versions))[:2]
    if len(targets) < 2:
        shorthand = re.search(
            r"\b([A-Za-z][A-Za-z0-9+.-]*)\s+(\d{1,2}(?:\.\d{1,2}){0,2})\s+"
            r"(?:vs\.?|versus|oder|or)\s+(\d{1,2}(?:\.\d{1,2}){0,2})\b", query, re.I)
        if shorthand:
            targets = (f"{shorthand[1]} {shorthand[2]}", f"{shorthand[1]} {shorthand[3]}")
    if len(targets) < 2:
        split = re.search(r"\b([\w.+-]+)\s+(?:vs\.?|versus|oder|or)\s+([\w.+-]+)\b", query, re.I)
        if split:
            targets = (split[1], split[2])
    comparison = len(targets) == 2
    mode = (ResearchMode.CURRENT_COMPARISON if comparison and scope != TemporalScope.TIMELESS else
            ResearchMode.COMPARISON if comparison else ResearchMode.CURRENT_STATE
            if scope != TemporalScope.TIMELESS else ResearchMode.FACTUAL)
    goal = (f"Compare {targets[0]} and {targets[1]} for the user's stated criteria; address both targets."
            if comparison else f"Answer the user's actual question directly: {query[:220]}")
    location = None
    location_match = re.search(r"\b(?:in|für|for)\s+([A-ZÄÖÜ][\wÄÖÜäöüß -]+(?:,\s*[A-ZÄÖÜ][\wÄÖÜäöüß -]+)?)", query)
    if location_match:
        location = location_match[1].strip()[:100]
    dimensions = ("compatibility",) if comparison and re.search(
        r"\b(ai|ki|llms?|kompatib\w*|compatib\w*|cuda|pytorch|ollama)\b", q) else ()
    criteria: tuple[ComparisonCriterion, ...] = ()
    requirements: tuple[EvidenceRequirement, ...] = ()
    recommendation = comparison and bool(re.search(
        r"\b(empfehl\w*|besser|welches|welche|which|recommend|better|should i use|für|for)\b", q))
    if recommendation:
        context_terms = _comparison_context_terms(query, targets)
        criteria = (
            ComparisonCriterion("support_lifecycle", "Support/Lifecycle", "default", 1.0, True, True),
            ComparisonCriterion("requested_context_compatibility", "Requested-context compatibility",
                                "user_query", 1.0, True, True),
            ComparisonCriterion("current_limitations", "Current limitations", "default", .5, False, True),
        )
        built = []
        for target in targets:
            built.append(EvidenceRequirement(f"support_lifecycle:{target}", "support_lifecycle", target,
                                             ("support", "lifecycle"), "CURRENT", "lifecycle"))
        compatibility_terms = context_terms or ("compatibility",)
        for target in targets:
            built.append(EvidenceRequirement(f"requested_context_compatibility:{target}",
                                             "requested_context_compatibility", target,
                                             (*compatibility_terms, "compatibility"), "CURRENT",
                                             "compatibility"))
            built.append(EvidenceRequirement(f"current_limitations:{target}", "current_limitations",
                                             target, ("current", "limitations"), "CURRENT",
                                             "limitations"))
        requirements = tuple(built)
    return ResearchQuestionSpec(query, mode, scope, target_date, scope != TemporalScope.TIMELESS,
                                goal, targets, location, dimensions, criteria, requirements)


@dataclass
class ResearchEvidence:
    evidence_id: str
    title: str
    url: str
    content_excerpt: str
    retrieved_at: str | None
    published_at: str | None
    updated_at: str | None
    evidence_date: date | None
    source_type: str
    relevance: float
    temporal_status: TemporalStatus
    role: EvidenceRole = EvidenceRole.REJECTED
    selected: bool = False
    rejection_reason: str | None = None
    target_matches: tuple[str, ...] = ()
    originating_queries: tuple[str, ...] = ()
    originating_providers: tuple[str, ...] = ()
    originating_requirement_ids: tuple[str, ...] = ()
    source: ResearchSource | None = field(default=None, repr=False)
    fact_candidates: list[FactCandidate] = field(default_factory=list)
    facts: list[EvidenceFact] = field(default_factory=list)


@dataclass(frozen=True)
class FactCandidate:
    fact_id: str
    text: str
    source_evidence_id: str
    query_relevance: float
    target_relevance: float
    boilerplate_penalty: float
    target: str | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    text: str
    source_evidence_id: str
    subject: str | None
    valid_from: date | None
    valid_until: date | None
    fact_date: date | None
    temporal_basis: TemporalBasis
    temporal_status: TemporalStatus
    confidence: float
    target: str | None = None
    evidence_type: str = "generic"
    volatility: EvidenceVolatility = EvidenceVolatility.MEDIUM
    requirement_ids: tuple[str, ...] = ()
    criterion: str | None = None


def _terms(text: str) -> set[str]:
    return {x for x in re.findall(r"[\wäöüß-]{3,}", text.casefold()) if x not in _GENERIC}


def _relevance(spec: ResearchQuestionSpec, source: ResearchSource) -> tuple[float, tuple[str, ...]]:
    content = f"{source.title} {source.snippet} {source.text[:1600]}".casefold()
    matches = tuple(target for target in spec.comparison_targets if target.casefold() in content)
    query_terms = _terms(spec.original_query) - {x for target in spec.comparison_targets for x in _terms(target)}
    overlap = len(query_terms & _terms(content)) / max(1, min(len(query_terms), 6))
    if spec.comparison_targets:
        score = min(1.0, .18 * len(matches) + .30 * overlap)
    else:
        score = min(1.0, .25 + .65 * overlap) if overlap else 0.0
    return round(score, 3), matches


def _temporal_status(spec: ResearchQuestionSpec, temporal: TemporalContext,
                     explicit_dates: set[date], published: date | None,
                     updated: date | None) -> tuple[TemporalStatus, date | None]:
    if len(explicit_dates) > 1:
        return TemporalStatus.TEMPORAL_UNKNOWN, None
    evidence_date = next(iter(explicit_dates)) if explicit_dates else None
    if not spec.requires_fresh_evidence:
        return TemporalStatus.TEMPORALLY_COMPATIBLE, evidence_date or updated or published
    target = spec.target_date or temporal.local_date
    if evidence_date:
        if evidence_date == target:
            return TemporalStatus.CONFIRMED_CURRENT, evidence_date
        return (TemporalStatus.STALE if evidence_date < target else TemporalStatus.TEMPORAL_CONFLICT), evidence_date
    metadata_date = updated or published
    if metadata_date:
        days = (target - metadata_date).days
        window = 0 if spec.target_date is not None else 90
        if 0 <= days <= window:
            return TemporalStatus.TEMPORALLY_COMPATIBLE, metadata_date
        return (TemporalStatus.STALE if days > window else TemporalStatus.TEMPORAL_CONFLICT), metadata_date
    return TemporalStatus.TEMPORAL_UNKNOWN, None


_BOILERPLATE = ("inhaltsverzeichnis", "navigation", "sprachen", "languages", "werkzeuge",
                "artikel diskussion", "links bearbeiten", "cookie", "login", "sign in",
                "toggle contents", "table of contents", "jump to content", "edit source")
_FACT_VERBS = re.compile(r"\b(is|are|has|have|means|provides|supports?|released?|available|enables?|uses?|"
                         r"ist|sind|bedeutet|ermöglicht|unterstützt|wird|werden|beträgt|zeigt|"
                         r"vorhersage|forecast|temperatur|temperature|support|eol|"
                         r"supported|maintenance|wartung|end of life|lts)\b", re.I)
_WEATHER = re.compile(r"\b(wetter|weather|forecast|vorhersage|temperatur|temperature|heute|today)\b", re.I)
_MONTH_YEAR = re.compile(r"\b(20\d{2})-(0?[1-9]|1[0-2])\b")
_MONTH_NAME_YEAR = re.compile(r"\b([A-Za-zä]+)\s+(20\d{2})\b", re.I)
_LIFECYCLE = re.compile(r"\b(releas(?:e|ed)|veröffentlicht|support|supported|unterstützt|"
                        r"maintenance|wartung|end of life|eol|lts)\b", re.I)
_COMPATIBILITY = re.compile(r"\b(compatib\w*|kompatib\w*|requirements?|anforderungen|"
                            r"install\w*|treiber|driver|cuda|pytorch|ollama|llms?|ai|ki)\b", re.I)
_COMPATIBILITY_DIRECT = re.compile(r"\b(compatib\w*|kompatib\w*|requirements?|anforderungen|"
                                   r"install\w*|treiber|driver|cuda|pytorch|ollama)\b", re.I)
_LIMITATIONS = re.compile(r"\b(limitations?|limitations?|known issues?|unsupported|not supported|"
                          r"einschränk\w*|bekannte probleme|nicht unterstützt|experimental|preview)\b", re.I)


def _evidence_type(text: str) -> str:
    if _LIMITATIONS.search(text):
        return "limitations"
    if _COMPATIBILITY_DIRECT.search(text):
        return "compatibility"
    if _LIFECYCLE.search(text):
        return "lifecycle"
    if _COMPATIBILITY.search(text) and re.search(r"\b(works?|runs?|läuft|funktioniert|supports?|unterstützt)\b", text, re.I):
        return "compatibility"
    return "generic"


def _fact_volatility(text: str, evidence_type: str) -> EvidenceVolatility:
    if _WEATHER.search(text) or re.search(r"\b(stock|share price|aktienkurs|live score|breaking news)\b", text, re.I):
        return EvidenceVolatility.HIGH
    if evidence_type in {"compatibility", "lifecycle", "limitations"}:
        return EvidenceVolatility.MEDIUM
    return EvidenceVolatility.LOW


def _authoritative_technical_source(source: ResearchSource, target: str) -> bool:
    parsed = urlparse(source.url)
    hostname = (parsed.hostname or "").casefold()
    product = target.split()[0].casefold().replace("+", "")
    product_alias = "postgres" if product == "postgresql" else product
    target_host = product_alias and product_alias in hostname
    documentation = (hostname.startswith(("docs.", "documentation.")) or
                     (bool(re.search(r"/(?:docs?|documentation|manual)(?:/|$)", parsed.path, re.I)) and
                      bool(re.search(r"\b(docs?|documentation|manual)\b", source.title, re.I))))
    return bool(target_host or documentation or
                any(domain in hostname for domain in ("ubuntu.com", "canonical.com", "python.org",
                                                       "postgresql.org")))


def _requirement_matches(spec: ResearchQuestionSpec, target: str | None,
                         evidence_type: str, text: str) -> tuple[tuple[str, ...], str | None]:
    if not target or target.casefold() not in text.casefold():
        return (), None
    matched = []
    criterion = None
    for requirement in spec.evidence_requirements:
        if requirement.target != target or requirement.evidence_type != evidence_type:
            continue
        if evidence_type == "lifecycle" and _LIFECYCLE.search(text):
            matched.append(requirement.requirement_id); criterion = requirement.criterion
        elif evidence_type == "compatibility":
            context = set(requirement.query_terms) - {"compatibility"}
            if context & _terms(text) or _COMPATIBILITY_DIRECT.search(text) or _COMPATIBILITY.search(text):
                matched.append(requirement.requirement_id); criterion = requirement.criterion
        elif evidence_type == "limitations" and _LIMITATIONS.search(text):
            matched.append(requirement.requirement_id); criterion = requirement.criterion
    return tuple(matched), criterion


def _month_bounds(text: str, *, end: bool = False) -> date | None:
    match = _MONTH_YEAR.search(text)
    if match:
        year, month = int(match[1]), int(match[2])
    else:
        named = _MONTH_NAME_YEAR.search(text)
        if not named or named[1].casefold() not in _MONTHS:
            return None
        year, month = int(named[2]), _MONTHS[named[1].casefold()]
    start = date(year, month, 1)
    if not end:
        return start
    next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return next_month - timedelta(days=1)


def _validity_interval(text: str) -> tuple[date | None, date | None]:
    release = re.search(r"\b(?:released?|release date|veröffentlicht|freigegeben|valid from|gültig ab)\b.{0,35}", text, re.I)
    support = re.search(r"\b(?:support(?:ed)? until|supported through|support ends?|"
                        r"(?:standard |maintenance )?support\s*:|end of (?:standard )?support|"
                        r"unterstützt bis|support bis|gültig bis|valid until|eol|maintenance until|"
                        r"wartung bis|end of life).{0,40}", text, re.I)
    release_dates = extract_dates(release[0]) if release else set()
    support_dates = extract_dates(support[0]) if support else set()
    valid_from = (next(iter(release_dates)) if len(release_dates) == 1 else
                  _month_bounds(release[0]) if release else None)
    valid_until = (next(iter(support_dates)) if len(support_dates) == 1 else
                   _month_bounds(support[0], end=True) if support else None)
    if support and not valid_until:
        year = re.search(r"\b(20\d{2})\b", support[0])
        # "until 2029" guarantees no more than the start of that year.
        valid_until = date(int(year[1]), 1, 1) if year else None
    return valid_from, valid_until


def _sentence_candidates(source: ResearchSource, spec: ResearchQuestionSpec,
                         evidence_id: str) -> list[FactCandidate]:
    pieces = re.split(r"\n+|(?<=[.!?])\s+(?=[A-ZÄÖÜ])", source.text[:5000])
    query_terms = _terms(spec.original_query)
    candidates: list[FactCandidate] = []
    for position, raw in enumerate(pieces[:80]):
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            continue
        if spec.comparison_targets and any(target.casefold() in text.casefold()
                                           for target in spec.comparison_targets):
            following = pieces[position + 1].strip() if position + 1 < len(pieces) else ""
            if (following and len(text) + len(following) < 260 and
                not any(target.casefold() in following.casefold() for target in spec.comparison_targets) and
                re.search(r"\b(support|unterstützt|gültig|valid|eol)\b", following, re.I)):
                text = f"{text} {following}"
        lower = text.casefold()
        boilerplate = any(phrase in lower for phrase in _BOILERPLATE)
        target = next((value for value in spec.comparison_targets if value.casefold() in lower), None)
        overlap = len(query_terms & _terms(text)) / max(1, min(len(query_terms), 8))
        target_score = 1.0 if target else 0.0
        reason = None
        if boilerplate:
            reason = "boilerplate"
        weather_measure = bool(re.search(r"-?\d{1,2}\s*°\s*C", text, re.I) and _WEATHER.search(text))
        criterion_fact = bool(target and _FACT_VERBS.search(text) and
                              (_LIFECYCLE.search(text) or _COMPATIBILITY.search(text) or
                               _LIMITATIONS.search(text)))
        elif_condition = ((len(text) < 25 or len(_terms(text)) < 3 or not _FACT_VERBS.search(text))
                          and not weather_measure and not criterion_fact)
        if not reason and elif_condition:
            reason = "not_a_fact_sentence"
        if not reason and spec.comparison_targets and not target:
            reason = "no_comparison_target"
        elif not reason and not spec.comparison_targets and overlap == 0:
            reason = "low_query_relevance"
        if len(candidates) >= 24:
            break
        candidates.append(FactCandidate(f"{evidence_id}_candidate_{len(candidates)+1}", text[:260], evidence_id,
                                        round(overlap, 3), target_score, 1.0 if boilerplate else 0.0,
                                        target, reason))
    candidates.sort(key=lambda x: (x.rejection_reason is None, x.target_relevance,
                                   x.query_relevance), reverse=True)
    return candidates


def _fact_validity(candidate: FactCandidate, source: ResearchSource,
                   spec: ResearchQuestionSpec, temporal: TemporalContext) -> EvidenceFact:
    text = candidate.text
    target = spec.target_date or temporal.local_date
    valid_from, valid_until = _validity_interval(text)
    fact_dates = extract_dates(text)
    basis = TemporalBasis.UNKNOWN
    fact_date = next(iter(fact_dates)) if len(fact_dates) == 1 else None
    status = TemporalStatus.TEMPORAL_UNKNOWN
    evidence_type = _evidence_type(text)
    if not spec.requires_fresh_evidence:
        basis, status = TemporalBasis.TIMELESS_FACT, TemporalStatus.TEMPORALLY_COMPATIBLE
    elif valid_until and (valid_from is None or valid_from <= valid_until):
        basis = TemporalBasis.VALIDITY_INTERVAL
        status = (TemporalStatus.TEMPORALLY_COMPATIBLE if
                  (valid_from is None or valid_from <= target) and target <= valid_until else
                  TemporalStatus.STALE if target > valid_until else TemporalStatus.TEMPORAL_CONFLICT)
    elif fact_date:
        basis = TemporalBasis.EXPLICIT_DATE
        status = (TemporalStatus.CONFIRMED_CURRENT if fact_date == target else
                  TemporalStatus.STALE if fact_date < target else TemporalStatus.TEMPORAL_CONFLICT)
    elif len(fact_dates) > 1:
        basis = TemporalBasis.UNKNOWN
    else:
        body_dates = extract_dates(source.text[:5000])
        markers = {_metadata_date(value) for value in source.time_markers}
        markers.discard(None)
        current_marker = (len(markers) == 1 and target in markers) or (len(body_dates) == 1 and target in body_dates)
        historical_marker = any(value < target for value in body_dates | markers)
        if _WEATHER.search(text) and current_marker and not historical_marker:
            basis, status = TemporalBasis.CURRENT_PAGE_MARKER, TemporalStatus.CONFIRMED_CURRENT
            fact_date = target
        elif historical_marker and _WEATHER.search(text):
            basis, status = TemporalBasis.EXPLICIT_DATE, TemporalStatus.STALE
            fact_date = min(body_dates | markers)
        else:
            metadata_date = _metadata_date(source.updated_at) or _metadata_date(source.published_at)
            window = 0 if spec.target_date else 90
            if metadata_date and 0 <= (target - metadata_date).days <= window:
                basis, status = TemporalBasis.STRUCTURED_METADATA, TemporalStatus.TEMPORALLY_COMPATIBLE
                fact_date = metadata_date
    requirement_ids, criterion = _requirement_matches(spec, candidate.target, evidence_type, text)
    volatility = _fact_volatility(text, evidence_type)
    explicit_conflict = status in {TemporalStatus.STALE, TemporalStatus.TEMPORAL_CONFLICT}
    if (spec.requires_fresh_evidence and status == TemporalStatus.TEMPORAL_UNKNOWN and
            volatility != EvidenceVolatility.HIGH and candidate.target and
            candidate.target.casefold() in text.casefold() and requirement_ids and
            _authoritative_technical_source(source, candidate.target) and not explicit_conflict):
        basis, status = (TemporalBasis.LIVE_AUTHORITATIVE_DOCUMENTATION,
                         TemporalStatus.TEMPORALLY_COMPATIBLE)
    confidence = min(1.0, .35 + candidate.query_relevance * .3 + candidate.target_relevance * .15 +
                     (.2 if basis in {TemporalBasis.EXPLICIT_DATE, TemporalBasis.VALIDITY_INTERVAL,
                                      TemporalBasis.CURRENT_PAGE_MARKER} else .1))
    return EvidenceFact(candidate.fact_id.replace("candidate", "fact"), text, candidate.source_evidence_id,
                        candidate.target or (source.title[:80] if source.title else None), valid_from, valid_until,
                        fact_date, basis, status, round(confidence, 3), candidate.target,
                        evidence_type, volatility, requirement_ids, criterion)


def extract_evidence_facts(source: ResearchSource, spec: ResearchQuestionSpec,
                           temporal: TemporalContext, evidence_id: str) -> tuple[list[FactCandidate], list[EvidenceFact]]:
    candidates = _sentence_candidates(source, spec, evidence_id)
    facts = [_fact_validity(candidate, source, spec, temporal)
             for candidate in candidates if candidate.rejection_reason is None]
    facts.sort(key=lambda item: (item.temporal_status in
               {TemporalStatus.CONFIRMED_CURRENT, TemporalStatus.TEMPORALLY_COMPATIBLE},
               item.confidence), reverse=True)
    return candidates, facts[:8]


def normalize_evidence(source: ResearchSource, index: int, spec: ResearchQuestionSpec,
                       temporal: TemporalContext) -> ResearchEvidence:
    published, updated = _metadata_date(source.published_at), _metadata_date(source.updated_at)
    # Scan the bounded fetched page; retrieval time is never an evidence date.
    explicit = extract_dates(f"{source.title} {source.snippet} {source.text[:5000]}")
    status, evidence_date = _temporal_status(spec, temporal, explicit, published, updated)
    relevance, matches = _relevance(spec, source)
    evidence_id = f"research_{index}"
    candidates, facts = extract_evidence_facts(source, spec, temporal, evidence_id) if source.ok and source.text else ([], [])
    return ResearchEvidence(evidence_id, source.title[:180], source.url,
                            (source.snippet or source.text)[:600], source.retrieved_at,
                            source.published_at, source.updated_at, evidence_date,
                            source.source_type, relevance, status, target_matches=matches,
                            originating_queries=source.originating_queries,
                            originating_providers=source.originating_providers, source=source,
                            originating_requirement_ids=source.originating_requirement_ids,
                            fact_candidates=candidates, facts=facts)


def select_evidence(spec: ResearchQuestionSpec, evidence: list[ResearchEvidence]) -> list[ResearchEvidence]:
    for item in evidence:
        if not item.source or not item.source.ok or not item.source.text:
            item.rejection_reason = "unreadable"; continue
        if spec.comparison_targets and not item.target_matches:
            item.role = EvidenceRole.SUPPORTING if item.relevance >= .1 else EvidenceRole.REJECTED
            item.rejection_reason = "no_comparison_target"; continue
        usable_facts = [fact for fact in item.facts if fact.temporal_status in
                        {TemporalStatus.CONFIRMED_CURRENT, TemporalStatus.TEMPORALLY_COMPATIBLE}]
        if spec.requires_fresh_evidence and not usable_facts:
            item.rejection_reason = (
                "source_explicit_temporal_conflict" if item.temporal_status in
                {TemporalStatus.STALE, TemporalStatus.TEMPORAL_CONFLICT} else
                "source_temporal_unknown_but_no_valid_fact" if
                item.temporal_status == TemporalStatus.TEMPORAL_UNKNOWN else "no_valid_fact")
            continue
        if spec.comparison_targets:
            if item.relevance < .16:
                item.rejection_reason = "no_query_relevance"; continue
        elif item.relevance < .20:
            item.rejection_reason = "no_query_relevance"; continue
        if not usable_facts:
            item.rejection_reason = "no_valid_fact"; continue
        item.role = EvidenceRole.PRIMARY
        item.selected = True
    selected = [e for e in evidence if e.selected]
    selected.sort(key=lambda e: (e.relevance + ((.25 if spec.comparison_targets else .08)
                             if urlparse(e.url).hostname and any(x in urlparse(e.url).hostname
                             for x in ("ubuntu.com", "canonical.com")) else 0),
                             e.temporal_status == TemporalStatus.CONFIRMED_CURRENT), reverse=True)
    prioritized: list[ResearchEvidence] = []
    for target in spec.comparison_targets:
        first = next((item for item in selected if target in item.target_matches and item not in prioritized), None)
        if first:
            prioritized.append(first)
    prioritized.extend(item for item in selected if item not in prioritized)
    # At most one source per required target/criterion plus bounded supporting room.
    selection_limit = min(6, max(4, len(spec.evidence_requirements)))
    keep = {e.evidence_id for e in prioritized[:selection_limit]}
    for item in selected:
        if item.evidence_id in keep:
            continue
        item.selected = False; item.role = EvidenceRole.REJECTED; item.rejection_reason = "selection_limit"
    return [e for e in prioritized if e.evidence_id in keep]


@dataclass(frozen=True)
class ResearchFact:
    fact_id: str
    fact: str
    source_evidence_ids: tuple[str, ...]
    temporal_status: TemporalStatus
    confidence: float
    target: str | None = None
    temporal_basis: TemporalBasis = TemporalBasis.UNKNOWN
    valid_from: date | None = None
    valid_until: date | None = None
    fact_date: date | None = None
    evidence_type: str = "generic"
    requirement_ids: tuple[str, ...] = ()
    criterion: str | None = None


@dataclass
class ResearchResponsePlan:
    response_mode: str
    answer_goal: str
    temporal_context: TemporalContext
    comparison_targets: tuple[str, ...]
    required_facts: list[ResearchFact]
    supporting_facts: list[ResearchFact]
    source_ids: list[str]
    required_caveats: list[str]
    forbidden_claims: list[str]
    confidence: float
    evidence_conflict: bool = False
    comparison_missing_reasons: dict[str, str] = field(default_factory=dict)
    comparison_criterion_coverage: float = 1.0
    comparison_criteria: tuple[ComparisonCriterion, ...] = ()
    evidence_requirements: tuple[EvidenceRequirement, ...] = ()
    requirement_coverage: dict[str, float] = field(default_factory=dict)
    recommendation_confidence: RecommendationConfidence = RecommendationConfidence.INSUFFICIENT

    def prompt(self, selected: list[ResearchEvidence]) -> str:
        lines = ["RESEARCH RESPONSE PLAN", "Answer the user's actual question directly.",
                 "Use required facts. Respect the temporal context. Do not use rejected or stale evidence for current claims.",
                 "Do not replace a requested comparison with a general topic summary.",
                 "Do not include a source list in the answer; selected sources are returned separately.",
                 f"response_mode: {self.response_mode}", f"answer_goal: {self.answer_goal}",
                 f"CURRENT DATE: {self.temporal_context.local_date.isoformat()} ({self.temporal_context.local_timezone})",
                 f"target_date: {self.temporal_context.target_date.isoformat() if self.temporal_context.target_date else '-'}",
                 "Do not turn partial evidence into a stronger recommendation. Explicitly distinguish missing evidence.",
                 f"COMPARISON TARGETS: {' | '.join(self.comparison_targets) or '-'}",
                 "EVIDENCE CRITERIA:"]
        lines += [f"- {c.criterion_id}: {c.label}" for c in self.comparison_criteria]
        lines += ["FACTS BY TARGET AND CRITERION:"]
        lines += [f"- {f.target or '-'} | {f.criterion or f.evidence_type}: "
                  f"[{','.join(f.source_evidence_ids)} | {f.temporal_basis.value}] {f.fact}"
                  for f in self.required_facts]
        missing = [r for r in self.evidence_requirements if r.required and not r.satisfied]
        lines += ["MISSING EVIDENCE:"] + [f"- {r.target} | {r.criterion}" for r in missing]
        lines += [f"RECOMMENDATION CONFIDENCE: {self.recommendation_confidence.value}"]
        lines += ["SUPPORTING FACTS:"] + [f"- {f.fact}" for f in self.supporting_facts]
        lines += ["REQUIRED CAVEATS:"] + [f"- {c}" for c in self.required_caveats]
        lines += ["FORBIDDEN CLAIMS:"] + [f"- {c}" for c in self.forbidden_claims]
        lines += ["SELECTED SOURCES:"]
        for evidence in selected:
            lines.append(f"- [{evidence.evidence_id}] {evidence.title} | {evidence.url} | {evidence.temporal_status.value}")
        return "\n".join(lines)

    def compliance(self, answer: str) -> dict:
        lower = answer.casefold()
        if self.response_mode == "insufficient_current_evidence":
            cautious = any(x in lower for x in ("keine ausreichend aktuelle", "nicht genug aktuelle",
                                                 "insufficient current", "not enough current"))
            return {"compliant": cautious, "required_fact_coverage": 1.0 if cautious else 0.0,
                    "comparison_target_coverage": 0.0, "temporal_claim_conflict": not cautious,
                    "forbidden_claim_conflict": not cautious}
        target_hits = sum(target.casefold() in lower for target in self.comparison_targets)
        target_coverage = target_hits / len(self.comparison_targets) if self.comparison_targets else 1.0
        target_terms = set().union(*(_terms(target) for target in self.comparison_targets)) if self.comparison_targets else set()
        fact_hits = sum(any(token in lower for token in (_terms(f.fact) - target_terms) if len(token) >= 5)
                        for f in self.required_facts)
        fact_coverage = fact_hits / len(self.required_facts) if self.required_facts else 0.0
        current_claim = bool(re.search(r"\b(heute|today|aktuell|current|derzeit)\b", lower))
        temporal_conflict = current_claim and any(f.temporal_status in
            {TemporalStatus.STALE, TemporalStatus.TEMPORAL_CONFLICT} for f in self.required_facts)
        if self.temporal_context.target_date and current_claim:
            answer_dates = extract_dates(answer)
            temporal_conflict = temporal_conflict or any(
                item != self.temporal_context.target_date for item in answer_dates)
        supported_temperatures = {int(m[1]) for fact in self.required_facts
            for m in re.finditer(r"(-?\d{1,2})\s*°\s*C", fact.fact, re.I)}
        claimed_temperatures = {int(m[1]) for m in re.finditer(r"(-?\d{1,2})\s*°\s*C", answer, re.I)}
        if supported_temperatures and claimed_temperatures - supported_temperatures:
            temporal_conflict = True
        forbidden_conflict = temporal_conflict
        compliant = fact_coverage >= 1.0 and target_coverage == 1.0 and not temporal_conflict
        if self.comparison_targets and target_coverage < 1.0:
            compliant = False
        if self.required_caveats and self.comparison_targets and any("fehlt belastbare" in c for c in self.required_caveats):
            caveat_present = any(x in lower for x in ("keine ausreichend", "fehlt", "insufficient", "not enough"))
            compliant = compliant and caveat_present
            if re.search(r"\b(besser|empfehle|empfehlung|recommend|prefer|best)\b", lower):
                compliant = False
                forbidden_conflict = True
        return {"compliant": compliant, "required_fact_coverage": fact_coverage,
                "comparison_target_coverage": target_coverage,
                "temporal_claim_conflict": temporal_conflict,
                "forbidden_claim_conflict": forbidden_conflict}

    def render(self) -> str:
        if self.response_mode == "insufficient_current_evidence":
            return "Ich habe keine ausreichend aktuelle Evidenz gefunden, um diese Frage zuverlässig als heutigen Stand zu beantworten."
        if self.comparison_targets:
            lines = ["Aus der ausgewählten Evidenz ergibt sich:"]
            for target in self.comparison_targets:
                lines.append(f"\n{target}")
                for criterion in self.comparison_criteria or (ComparisonCriterion("evidence", "Evidence", "", 1, True, True),):
                    facts = [f.fact for f in self.required_facts
                             if f.target == target and (f.criterion == criterion.criterion_id or
                                                        not self.comparison_criteria)]
                    lines.append(f"- {criterion.label}: " +
                                 (facts[0] if facts else "Dafür liegt keine ausreichend belastbare Evidenz vor."))
            if self.required_caveats:
                lines.append("Einschränkung: " + " ".join(self.required_caveats[:2]))
            lines.append(f"Fazit: Empfehlungssicherheit {self.recommendation_confidence.value}.")
            return "\n".join(lines)
        facts = [f.fact for f in self.required_facts[:2]]
        if not facts:
            return "Ich habe Quellen gefunden, konnte daraus aber keine ausreichend klare belegte Antwort extrahieren."
        answer = "Aus der ausgewählten Evidenz: " + " ".join(facts)
        if self.required_caveats:
            answer += " Einschränkung: " + " ".join(self.required_caveats[:2])
        return answer


def build_research_plan(spec: ResearchQuestionSpec, temporal: TemporalContext,
                        selected: list[ResearchEvidence],
                        all_evidence: list[ResearchEvidence] | None = None) -> ResearchResponsePlan:
    targets = spec.comparison_targets
    valid_facts = [(item, fact) for item in selected for fact in item.facts if fact.temporal_status in
                   {TemporalStatus.CONFIRMED_CURRENT, TemporalStatus.TEMPORALLY_COMPATIBLE}]
    covered = {fact.target for _, fact in valid_facts if fact.target}
    source_pool = all_evidence if all_evidence is not None else selected
    missing_reasons = {target: ("sources_found_but_no_valid_facts" if any(
        target in item.target_matches for item in source_pool) else "source_missing")
        for target in targets if target not in covered}
    fresh_enough = bool(valid_facts)
    insufficient = (spec.requires_fresh_evidence and not fresh_enough)
    caveats = []
    if targets and set(targets) != covered:
        caveats.append("Für einen vollständigen Vergleich fehlt belastbare Evidenz zu: " +
                       ", ".join(target for target in targets if target not in covered) + ".")
    fact_by_requirement = {requirement.requirement_id: [fact for _, fact in valid_facts
                           if requirement.requirement_id in fact.requirement_ids]
                           for requirement in spec.evidence_requirements}
    planned_requirements = tuple(replace(requirement, satisfied=bool(fact_by_requirement[requirement.requirement_id]),
                                         supporting_fact_ids=tuple(f.fact_id for f in
                                                                   fact_by_requirement[requirement.requirement_id][:3]))
                                 for requirement in spec.evidence_requirements)
    criterion_coverages = {}
    for criterion in spec.comparison_criteria:
        relevant = [requirement for requirement in planned_requirements
                    if requirement.criterion == criterion.criterion_id and requirement.required]
        criterion_coverages[criterion.criterion_id] = (
            sum(requirement.satisfied for requirement in relevant) / len(relevant) if relevant else 0.0)
    criterion_requested = any(c.criterion_id == "requested_context_compatibility"
                              for c in spec.comparison_criteria)
    criterion_coverage = criterion_coverages.get("requested_context_compatibility",
                                                  1.0 if not criterion_requested else 0.0)
    if targets and criterion_requested and criterion_coverage < 1.0:
        caveats.append("Für die angefragte Kompatibilität fehlt belastbare Evidenz für eine Empfehlung; "
                       "Lifecycle-Daten allein belegen keine Eignung.")
    if insufficient:
        caveats.append("Keine ausreichend aktuelle Evidenz ausgewählt.")
    facts: list[ResearchFact] = []
    if targets and planned_requirements:
        used = set()
        for requirement in planned_requirements:
            match = next(((item, fact) for item, fact in valid_facts
                          if fact.fact_id not in used and
                          requirement.requirement_id in fact.requirement_ids), None)
            if match:
                item, fact = match
                used.add(fact.fact_id)
                facts.append(ResearchFact(fact.fact_id, fact.text, (item.evidence_id,),
                                          fact.temporal_status, fact.confidence, fact.target,
                                          fact.temporal_basis, fact.valid_from, fact.valid_until,
                                          fact.fact_date, fact.evidence_type, fact.requirement_ids,
                                          fact.criterion))
    elif targets:
        for target in targets:
            target_matches = [(item, fact) for item, fact in valid_facts if fact.target == target]
            chosen = target_matches[:1]
            if criterion_requested:
                criterion_match = next(((item, fact) for item, fact in target_matches
                                        if fact.evidence_type == "compatibility"), None)
                if criterion_match and criterion_match not in chosen:
                    chosen.append(criterion_match)
            for match in chosen:
                item, fact = match
                facts.append(ResearchFact(fact.fact_id, fact.text, (item.evidence_id,),
                                          fact.temporal_status, fact.confidence, target,
                                          fact.temporal_basis, fact.valid_from, fact.valid_until, fact.fact_date,
                                          fact.evidence_type, fact.requirement_ids, fact.criterion))
    else:
        for item, fact in valid_facts[:2]:
            facts.append(ResearchFact(fact.fact_id, fact.text, (item.evidence_id,),
                                      fact.temporal_status, fact.confidence, None,
                                      fact.temporal_basis, fact.valid_from, fact.valid_until, fact.fact_date,
                                      fact.evidence_type, fact.requirement_ids, fact.criterion))
    mode = ("insufficient_current_evidence" if insufficient else "comparison_answer" if targets else
            "current_state_answer" if spec.requires_fresh_evidence else "direct_answer")
    values = set()
    for item in selected:
        match = re.search(r"(-?\d{1,2})\s*°\s*C", item.content_excerpt, re.I)
        if match: values.add(int(match[1]))
    conflict = len(values) > 1 and max(values) - min(values) >= 3
    if conflict:
        caveats.append("Ausgewählte aktuelle Quellen nennen widersprüchliche Werte.")
    forbidden = ["Do not present stale evidence as current", "Do not infer today's value from historical data"]
    if targets: forbidden.append("Do not recommend one target without evidence for the stated criterion")
    fact_source_ids = list(dict.fromkeys(source_id for fact in facts for source_id in fact.source_evidence_ids))
    required_requirements = [r for r in planned_requirements if r.required]
    overall_requirement_coverage = (sum(r.satisfied for r in required_requirements) /
                                    len(required_requirements) if required_requirements else 1.0)
    recommendation_confidence = (RecommendationConfidence.HIGH if overall_requirement_coverage == 1.0 else
                                 RecommendationConfidence.MEDIUM if overall_requirement_coverage >= .75 else
                                 RecommendationConfidence.LOW if overall_requirement_coverage >= .5 else
                                 RecommendationConfidence.INSUFFICIENT)
    return ResearchResponsePlan(
        response_mode=mode, answer_goal=spec.answer_goal, temporal_context=temporal,
        comparison_targets=targets, required_facts=facts, supporting_facts=[],
        source_ids=fact_source_ids, required_caveats=caveats, forbidden_claims=forbidden,
        confidence=min((e.relevance for e in selected), default=0.0), evidence_conflict=conflict,
        comparison_missing_reasons=missing_reasons,
        comparison_criterion_coverage=criterion_coverage,
        comparison_criteria=spec.comparison_criteria, evidence_requirements=planned_requirements,
        requirement_coverage=criterion_coverages,
        recommendation_confidence=recommendation_confidence)


def research_debug(spec: ResearchQuestionSpec, temporal: TemporalContext,
                   evidence: list[ResearchEvidence], plan: ResearchResponsePlan,
                   compliance: dict, fallback_used: bool) -> dict:
    counts = Counter(e.temporal_status.value for e in evidence)
    covered = {fact.target for fact in plan.required_facts if fact.target}
    source_covered = {target for item in evidence if item.selected for target in item.target_matches}
    candidates = [candidate for item in evidence for candidate in item.fact_candidates]
    selected_facts = [fact for item in evidence if item.selected for fact in item.facts
                      if fact.temporal_status in {TemporalStatus.CONFIRMED_CURRENT,
                                                  TemporalStatus.TEMPORALLY_COMPATIBLE}]
    selected_fact_ids = [fact.fact_id for fact in plan.required_facts]
    selected_fact_id_set = set(selected_fact_ids)
    target_source_counts = {target: sum(target in item.target_matches for item in evidence)
                            for target in spec.comparison_targets}
    target_fact_counts = {target: sum(fact.target == target for fact in selected_facts)
                          for target in spec.comparison_targets}
    return {"research_mode": spec.research_mode.value, "temporal_scope": spec.temporal_scope.value,
            "current_date": temporal.local_date.isoformat(),
            "target_date": spec.target_date.isoformat() if spec.target_date else None,
            "requires_fresh_evidence": spec.requires_fresh_evidence,
            "results_seen": len(evidence), "evidence_selected": sum(e.selected for e in evidence),
            "evidence_rejected": sum(not e.selected for e in evidence),
            "selected_evidence_ids": [e.evidence_id for e in evidence if e.selected],
            "rejected_evidence_ids": [e.evidence_id for e in evidence if not e.selected],
            "rejection_reasons": {e.evidence_id: e.rejection_reason for e in evidence if not e.selected},
            "temporal_status_counts": dict(counts), "comparison_targets": list(spec.comparison_targets),
            "comparison_target_coverage": len(covered) / len(spec.comparison_targets) if spec.comparison_targets else 1.0,
            "comparison_fact_coverage": len(covered) / len(spec.comparison_targets) if spec.comparison_targets else 1.0,
            "comparison_source_coverage": len(source_covered) / len(spec.comparison_targets) if spec.comparison_targets else 1.0,
            "comparison_missing_reasons": dict(plan.comparison_missing_reasons),
            "comparison_criterion_coverage": plan.comparison_criterion_coverage,
            "comparison_criteria": [criterion.criterion_id for criterion in plan.comparison_criteria],
            "evidence_requirement_count": len(plan.evidence_requirements),
            "evidence_requirements_satisfied": sum(r.satisfied for r in plan.evidence_requirements),
            "requirement_coverage": dict(plan.requirement_coverage),
            "missing_requirement_ids": [r.requirement_id for r in plan.evidence_requirements
                                        if r.required and not r.satisfied],
            "recommendation_confidence": plan.recommendation_confidence.value,
            "comparison_target_fact_counts": target_fact_counts,
            "target_source_counts": target_source_counts,
            "inspection_evidence_count": len(evidence),
            "facts_rescued_from_temporal_unknown": sum(
                item.temporal_status == TemporalStatus.TEMPORAL_UNKNOWN for item in evidence
                for fact in item.facts if item.selected and fact.fact_id in selected_fact_id_set),
            "facts_rescued_from_stale_source": sum(
                item.temporal_status == TemporalStatus.STALE for item in evidence
                for fact in item.facts if item.selected and fact.fact_id in selected_fact_id_set),
            "matched_comparison_targets": {item.evidence_id: list(item.target_matches)
                                           for item in evidence[:8]},
            "originating_queries": {item.evidence_id: list(item.originating_queries)
                                    for item in evidence[:8]},
            "originating_providers": {item.evidence_id: list(item.originating_providers)
                                      for item in evidence[:8]},
            "originating_requirement_ids": {item.evidence_id: list(item.originating_requirement_ids)
                                             for item in evidence[:8]},
            "fact_candidates_seen": len(candidates), "facts_validated": len(selected_facts),
            "facts_rejected": len(candidates) - len(selected_facts),
            "fact_temporal_basis_counts": dict(Counter(fact.temporal_basis.value for fact in selected_facts)),
            "fact_volatility_counts": dict(Counter(fact.volatility.value for fact in selected_facts)),
            "selected_fact_ids": selected_fact_ids[:8],
            "rejected_fact_ids": [candidate.fact_id for candidate in candidates
                                  if candidate.fact_id.replace("candidate", "fact") not in selected_fact_id_set][:12],
            "response_plan_active": True, "response_mode": plan.response_mode,
            "required_fact_count": len(plan.required_facts),
            "required_fact_coverage": compliance["required_fact_coverage"],
            "temporal_claim_conflict": compliance["temporal_claim_conflict"],
            "forbidden_claim_conflict": compliance["forbidden_claim_conflict"],
            "citation_conflict": compliance.get("citation_conflict", False),
            "evidence_conflict": plan.evidence_conflict,
            "plan_compliance": compliance["compliant"], "fallback_used": fallback_used}
