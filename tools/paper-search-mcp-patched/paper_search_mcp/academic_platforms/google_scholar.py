from __future__ import annotations

import logging
import random
import re
import time
from datetime import datetime
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from ..paper import Paper

logger = logging.getLogger(__name__)


class PaperSource:
    """Abstract base class for paper sources."""

    def search(self, query: str, **kwargs) -> List[Paper]:
        raise NotImplementedError

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError

    def read_paper(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError


class GoogleScholarSearcher(PaperSource):
    """Best-effort Google Scholar search implementation (HTML scraping)."""

    SCHOLAR_URL = "https://scholar.google.com/scholar"
    BROWSERS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    ]

    def __init__(self):
        self.timeout = 20
        self._setup_session()

    def _setup_session(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": random.choice(self.BROWSERS),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        match = re.search(r"\b(19\d{2}|20\d{2})\b", text or "")
        if not match:
            return None
        year = int(match.group(1))
        return year if 1900 <= year <= datetime.now().year else None

    @staticmethod
    def _extract_citations(text: str) -> int:
        match = re.search(r"Cited by\s+(\d+)", text or "", flags=re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def _parse_paper(self, item) -> Optional[Paper]:
        try:
            title_elem = item.find("h3", class_="gs_rt")
            info_elem = item.find("div", class_="gs_a")
            abstract_elem = item.find("div", class_="gs_rs")
            footer_elem = item.find("div", class_="gs_fl")

            if not title_elem or not info_elem:
                return None

            title = re.sub(r"^\s*\[(PDF|HTML)\]\s*", "", title_elem.get_text(" ", strip=True), flags=re.I)
            link = title_elem.find("a", href=True)
            url = link["href"] if link else ""

            info_text = info_elem.get_text(" ", strip=True)
            authors_segment = info_text.split("-", 1)[0]
            authors = [a.strip() for a in authors_segment.split(",") if a.strip()]
            year = self._extract_year(info_text)
            citations = self._extract_citations(footer_elem.get_text(" ", strip=True) if footer_elem else "")

            return Paper(
                paper_id=f"gs_{abs(hash(url or title))}",
                title=title,
                authors=authors,
                abstract=abstract_elem.get_text(" ", strip=True) if abstract_elem else "",
                url=url,
                pdf_url="",
                published_date=datetime(year, 1, 1) if year else datetime.utcnow(),
                updated_date=None,
                source="google_scholar",
                categories=[],
                keywords=[],
                doi="",
                citations=citations,
            )
        except Exception as e:
            logger.warning("Failed to parse paper: %s", e)
            return None

    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        papers: List[Paper] = []
        start = 0
        max_results = max(1, int(max_results))
        max_pages = max(1, (max_results + 9) // 10)

        for _ in range(max_pages):
            if len(papers) >= max_results:
                break

            params = {
                "q": query,
                "start": start,
                "hl": "en",
                "as_sdt": "0,5",
            }

            try:
                time.sleep(random.uniform(1.0, 2.0))
                response = self.session.get(self.SCHOLAR_URL, params=params, timeout=self.timeout)
                if response.status_code in {403, 429, 503}:
                    logger.warning("Google Scholar request blocked: status=%s", response.status_code)
                    break
                if response.status_code != 200:
                    logger.error("Search failed with status %s", response.status_code)
                    break

                lowered = response.text.lower()
                if "unusual traffic" in lowered or "not a robot" in lowered:
                    logger.warning("Google Scholar anti-bot page detected; stopping")
                    break

                soup = BeautifulSoup(response.text, "html.parser")
                results = soup.find_all("div", class_="gs_ri")
                if not results:
                    break

                added = 0
                for item in results:
                    if len(papers) >= max_results:
                        break
                    paper = self._parse_paper(item)
                    if paper:
                        papers.append(paper)
                        added += 1

                if added == 0:
                    break
                start += len(results)

            except requests.RequestException as e:
                logger.error("Search request error: %s", e)
                break
            except Exception as e:
                logger.error("Search error: %s", e)
                break

        return papers[:max_results]

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError(
            "Google Scholar does not provide direct PDF downloads. Use the paper URL to access full text."
        )

    def read_paper(self, paper_id: str, save_path: str = "./downloads") -> str:
        return (
            "Google Scholar does not support direct paper reading. "
            "Use the paper URL to access publisher-hosted full text."
        )
