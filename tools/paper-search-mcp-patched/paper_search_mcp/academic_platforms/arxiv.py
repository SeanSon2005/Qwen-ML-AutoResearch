from __future__ import annotations

import os
import re
from datetime import datetime
from typing import List, Optional

import feedparser
import requests
from PyPDF2 import PdfReader

from ..paper import Paper


class PaperSource:
    """Abstract base class for paper sources."""

    def search(self, query: str, **kwargs) -> List[Paper]:
        raise NotImplementedError

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError

    def read_paper(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError


class ArxivSearcher(PaperSource):
    """Searcher for arXiv papers."""

    BASE_URL = "https://export.arxiv.org/api/query"
    FIELD_QUERY_RE = re.compile(r"\b(all|ti|au|abs|cat|id|jr|rn|co|doi):", re.IGNORECASE)
    ARXIV_ID_RE = re.compile(
        r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$",
        re.IGNORECASE,
    )

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()

    def _normalize_query(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            return "all:*"

        # If caller already uses arXiv field syntax, keep as-is.
        if self.FIELD_QUERY_RE.search(q):
            return q

        # Fast path for exact arXiv ID lookups.
        if self.ARXIV_ID_RE.match(q):
            return f"id:{q}"

        escaped = q.replace('"', "")
        return f'all:"{escaped}"'

    @staticmethod
    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None

    def search(self, query: str, max_results: int = 10, sort_by: str = "relevance") -> List[Paper]:
        sort_key = sort_by if sort_by in {"relevance", "submittedDate", "lastUpdatedDate"} else "relevance"
        params = {
            "search_query": self._normalize_query(query),
            "max_results": max(1, int(max_results)),
            "sortBy": sort_key,
            "sortOrder": "descending",
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            return []

        feed = feedparser.parse(response.content)
        papers: List[Paper] = []
        for entry in getattr(feed, "entries", []):
            try:
                entry_id = getattr(entry, "id", "")
                paper_id = entry_id.split("/")[-1] if entry_id else ""
                authors = [getattr(author, "name", "").strip() for author in getattr(entry, "authors", [])]
                authors = [a for a in authors if a]
                published = self._parse_dt(getattr(entry, "published", None))
                updated = self._parse_dt(getattr(entry, "updated", None))
                tags = [getattr(tag, "term", "") for tag in getattr(entry, "tags", [])]
                tags = [t for t in tags if t]
                pdf_url = next(
                    (
                        getattr(link, "href", "")
                        for link in getattr(entry, "links", [])
                        if getattr(link, "type", "") == "application/pdf"
                    ),
                    "",
                )

                papers.append(
                    Paper(
                        paper_id=paper_id,
                        title=(getattr(entry, "title", "") or "").strip(),
                        authors=authors,
                        abstract=(getattr(entry, "summary", "") or "").strip(),
                        url=entry_id,
                        pdf_url=pdf_url,
                        published_date=published or datetime.utcnow(),
                        updated_date=updated,
                        source="arxiv",
                        categories=tags,
                        keywords=[],
                        doi=getattr(entry, "doi", "") or "",
                    )
                )
            except Exception:
                continue
        return papers

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        os.makedirs(save_path, exist_ok=True)
        safe_id = paper_id.replace("/", "_")
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        response = self.session.get(pdf_url, timeout=self.timeout)
        response.raise_for_status()

        output_file = os.path.join(save_path, f"{safe_id}.pdf")
        with open(output_file, "wb") as f:
            f.write(response.content)
        return output_file

    def read_paper(self, paper_id: str, save_path: str = "./downloads") -> str:
        os.makedirs(save_path, exist_ok=True)
        pdf_path = os.path.join(save_path, f"{paper_id.replace('/', '_')}.pdf")
        if not os.path.exists(pdf_path):
            pdf_path = self.download_pdf(paper_id, save_path)

        try:
            reader = PdfReader(pdf_path)
            text_chunks = []
            for page in reader.pages:
                text_chunks.append(page.extract_text() or "")
            return "\n".join(text_chunks).strip()
        except Exception:
            return ""
