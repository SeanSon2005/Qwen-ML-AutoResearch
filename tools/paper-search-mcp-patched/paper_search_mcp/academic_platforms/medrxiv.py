from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import List

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


class MedRxivSearcher(PaperSource):
    """Searcher for medRxiv papers."""

    BASE_URL = "https://api.biorxiv.org/details/medrxiv"

    def __init__(self):
        self.session = requests.Session()
        self.session.proxies = {"http": None, "https": None}
        self.timeout = 30
        self.max_retries = 3

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower()).strip()

    @staticmethod
    def _query_as_category(query: str) -> str:
        return query.strip().lower().replace(" ", "_")

    def _matches_keyword(self, query: str, item: dict) -> bool:
        q = self._normalize_text(query)
        haystack = self._normalize_text(
            " ".join(
                [
                    item.get("title", ""),
                    item.get("abstract", ""),
                    item.get("authors", ""),
                    item.get("category", ""),
                ]
            )
        )
        return q in haystack if q else True

    def _fetch_collection(self, url: str):
        tries = 0
        while tries < self.max_retries:
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                return data.get("collection", [])
            except requests.exceptions.RequestException:
                tries += 1
        return []

    def search(self, query: str, max_results: int = 10, days: int = 30) -> List[Paper]:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        raw_query = (query or "").strip()
        category_mode = raw_query.lower().startswith("category:")
        category = self._query_as_category(raw_query.split(":", 1)[1] if category_mode else raw_query)

        papers: List[Paper] = []
        cursor = 0
        max_results = max(1, int(max_results))

        while len(papers) < max_results:
            url = f"{self.BASE_URL}/{start_date}/{end_date}/{cursor}"
            if category_mode and category:
                url += f"?category={category}"

            collection = self._fetch_collection(url)
            if not collection:
                break

            for item in collection:
                try:
                    if not category_mode and raw_query and not self._matches_keyword(raw_query, item):
                        continue

                    date = datetime.strptime(item["date"], "%Y-%m-%d")
                    version = item.get("version", "1")
                    doi = item["doi"]
                    papers.append(
                        Paper(
                            paper_id=doi,
                            title=item.get("title", ""),
                            authors=(item.get("authors", "") or "").split("; "),
                            abstract=item.get("abstract", ""),
                            url=f"https://www.medrxiv.org/content/{doi}v{version}",
                            pdf_url=f"https://www.medrxiv.org/content/{doi}v{version}.full.pdf",
                            published_date=date,
                            updated_date=date,
                            source="medrxiv",
                            categories=[item.get("category", "")],
                            keywords=[],
                            doi=doi,
                        )
                    )
                    if len(papers) >= max_results:
                        break
                except Exception:
                    continue

            if len(papers) >= max_results or len(collection) < 100:
                break
            cursor += 100

        # Keyword searches over only 30 days are often sparse. Retry with a wider
        # window once before giving up.
        if not papers and not category_mode and days < 365:
            return self.search(raw_query, max_results=max_results, days=365)

        return papers[:max_results]

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        if not paper_id:
            raise ValueError("Invalid paper_id: paper_id is empty")

        os.makedirs(save_path, exist_ok=True)
        safe_id = paper_id.replace("/", "_")
        pdf_url = f"https://www.medrxiv.org/content/{paper_id}v1.full.pdf"

        tries = 0
        while tries < self.max_retries:
            try:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    )
                }
                response = self.session.get(pdf_url, timeout=self.timeout, headers=headers)
                response.raise_for_status()
                output_file = os.path.join(save_path, f"{safe_id}.pdf")
                with open(output_file, "wb") as f:
                    f.write(response.content)
                return output_file
            except requests.exceptions.RequestException as e:
                tries += 1
                if tries == self.max_retries:
                    raise Exception(f"Failed to download PDF after {self.max_retries} attempts: {e}")

        raise RuntimeError("Unreachable")

    def read_paper(self, paper_id: str, save_path: str = "./downloads") -> str:
        os.makedirs(save_path, exist_ok=True)
        pdf_path = os.path.join(save_path, f"{paper_id.replace('/', '_')}.pdf")
        if not os.path.exists(pdf_path):
            pdf_path = self.download_pdf(paper_id, save_path)

        try:
            reader = PdfReader(pdf_path)
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return "\n".join(text).strip()
        except Exception:
            return ""
