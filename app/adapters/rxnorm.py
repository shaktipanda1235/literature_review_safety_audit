from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from app.config import RXNAV_BASE
from app.sources.base import HttpClient


@dataclass
class RxNormResolution:
    """A matched RxNorm concept and its normalized ingredient names."""

    input_name: str
    rxcui: str
    canonical_name: str
    synonyms: List[str]
    ingredient_rxcuis: List[str]
    matched_name: str


class RxNormClient:
    """Resolve drug names through RxNav's approximate-term and related APIs."""

    def __init__(self, client: HttpClient | None = None, max_entries: int = 10):
        self.client = client or HttpClient()
        self._owns_client = client is None
        self.max_entries = max(1, min(max_entries, 100))
        self.last_error: str | None = None

    async def __aenter__(self) -> RxNormClient:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self.client.close()
            self._owns_client = False

    async def _get_json(self, url: str, params: Dict[str, Any] | None = None) -> Any:
        response = await self.client.request("GET", url, params=params or {}, use_cache=True)
        return response.json()

    @staticmethod
    def _list(value: Any) -> List[dict]:
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _text(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    async def resolve(self, drug_name: str) -> RxNormResolution | None:
        """Resolve a brand, generic, or misspelled name to its RxNorm ingredient."""
        self.last_error = None
        query = drug_name.strip()
        if not query:
            self.last_error = "Drug name is empty."
            return None

        try:
            approximate_payload = await self._get_json(
                f"{RXNAV_BASE}/approximateTerm.json",
                {"term": query, "maxEntries": self.max_entries, "option": 1},
            )
        except Exception as error:
            self.last_error = f"RxNav approximate-term lookup failed for '{query}': {error}"
            return None

        if not isinstance(approximate_payload, dict):
            self.last_error = f"RxNav returned malformed approximate-term data for '{query}'."
            return None

        approximate_group = approximate_payload.get("approximateGroup") or {}
        if not isinstance(approximate_group, dict):
            self.last_error = f"RxNav returned malformed approximate-term data for '{query}'."
            return None
        candidates = self._list(approximate_group.get("candidate"))
        candidates = [candidate for candidate in candidates if self._text(candidate.get("rxcui"))]
        if not candidates:
            self.last_error = f"No RxNorm match found for '{query}'."
            return None

        def candidate_order(candidate: dict) -> tuple[int, float, bool]:
            try:
                rank = int(candidate.get("rank", 10**9))
            except (TypeError, ValueError):
                rank = 10**9
            try:
                score = float(candidate.get("score", 0))
            except (TypeError, ValueError):
                score = 0
            return rank, -score, not bool(self._text(candidate.get("name")))

        candidate = min(candidates, key=candidate_order)
        matched_rxcui = self._text(candidate.get("rxcui"))
        try:
            related_payload = await self._get_json(
                f"{RXNAV_BASE}/rxcui/{matched_rxcui}/related.json", {"tty": "IN"}
            )
        except Exception as error:
            self.last_error = f"RxNav ingredient lookup failed for '{query}': {error}"
            return None

        if not isinstance(related_payload, dict):
            self.last_error = f"RxNav returned malformed ingredient data for '{query}'."
            return None

        related_group = related_payload.get("relatedGroup") or {}
        if not isinstance(related_group, dict):
            self.last_error = f"RxNav returned malformed ingredient data for '{query}'."
            return None
        ingredients = []
        for group in self._list(related_group.get("conceptGroup")):
            if group.get("tty") != "IN":
                continue
            ingredients.extend(self._list(group.get("conceptProperties")))

        ingredient_names = []
        ingredient_rxcuis = []
        synonyms = []

        def add_unique(values: List[str], value: Any) -> None:
            text = self._text(value)
            if text and text.casefold() not in {existing.casefold() for existing in values}:
                values.append(text)

        for ingredient in ingredients:
            add_unique(ingredient_names, ingredient.get("name"))
            add_unique(ingredient_rxcuis, ingredient.get("rxcui"))
            add_unique(synonyms, ingredient.get("name"))
            add_unique(synonyms, ingredient.get("synonym"))

        if not ingredient_names:
            self.last_error = f"RxNorm matched '{query}', but no active ingredient was found."
            return None

        matched_name = self._text(candidate.get("name")) or query
        add_unique(synonyms, matched_name)

        return RxNormResolution(
            input_name=query,
            rxcui=matched_rxcui,
            canonical_name=" + ".join(ingredient_names),
            synonyms=synonyms,
            ingredient_rxcuis=ingredient_rxcuis,
            matched_name=matched_name,
        )