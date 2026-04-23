from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional
from xml.etree import ElementTree as ET

import requests

from ..paper import Paper


class PaperSource:
    """Abstract base class for paper sources."""

    def search(self, query: str, **kwargs) -> List[Paper]:
        raise NotImplementedError

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError

    def read_paper(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError


class PubMedSearcher(PaperSource):
    """Searcher for PubMed papers."""

    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()

    @staticmethod
    def _text(elem: Optional[ET.Element]) -> str:
        if elem is None:
            return ""
        return "".join(elem.itertext()).strip()

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        match = re.search(r"\b(19\d{2}|20\d{2})\b", text or "")
        return int(match.group(1)) if match else None

    def _parse_pub_date(self, article: ET.Element) -> datetime:
        year_text = self._text(article.find('.//PubDate/Year'))
        if year_text.isdigit():
            return datetime(int(year_text), 1, 1)

        medline_date = self._text(article.find('.//PubDate/MedlineDate'))
        medline_year = self._extract_year(medline_date)
        if medline_year is not None:
            return datetime(medline_year, 1, 1)

        epub_year = self._text(article.find('.//ArticleDate/Year'))
        if epub_year.isdigit():
            return datetime(int(epub_year), 1, 1)

        return datetime.utcnow()

    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max(1, int(max_results)),
            "retmode": "xml",
            "sort": "relevance",
        }

        try:
            search_response = self.session.get(self.SEARCH_URL, params=search_params, timeout=self.timeout)
            search_response.raise_for_status()
            search_root = ET.fromstring(search_response.content)
        except Exception:
            return []

        ids = [node.text for node in search_root.findall('.//Id') if node is not None and node.text]
        if not ids:
            return []

        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        }

        try:
            fetch_response = self.session.get(self.FETCH_URL, params=fetch_params, timeout=self.timeout)
            fetch_response.raise_for_status()
            fetch_root = ET.fromstring(fetch_response.content)
        except Exception:
            return []

        papers: List[Paper] = []
        for article in fetch_root.findall('.//PubmedArticle'):
            try:
                pmid = self._text(article.find('.//PMID'))
                if not pmid:
                    continue

                title = self._text(article.find('.//ArticleTitle'))
                if not title:
                    continue

                authors: List[str] = []
                for author in article.findall('.//Author'):
                    collective = self._text(author.find('CollectiveName'))
                    if collective:
                        authors.append(collective)
                        continue

                    last = self._text(author.find('LastName'))
                    initials = self._text(author.find('Initials'))
                    full = " ".join(part for part in [last, initials] if part).strip()
                    if full:
                        authors.append(full)

                abstract_parts = []
                for a in article.findall('.//Abstract/AbstractText'):
                    txt = self._text(a)
                    if txt:
                        abstract_parts.append(txt)
                abstract = "\n".join(abstract_parts).strip()

                doi = self._text(article.find('.//ELocationID[@EIdType="doi"]'))
                if not doi:
                    doi = self._text(article.find('.//ArticleIdList/ArticleId[@IdType="doi"]'))

                published = self._parse_pub_date(article)

                papers.append(
                    Paper(
                        paper_id=pmid,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        pdf_url="",
                        published_date=published,
                        updated_date=published,
                        source="pubmed",
                        categories=[],
                        keywords=[],
                        doi=doi,
                    )
                )
            except Exception:
                continue

        return papers

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError(
            "PubMed does not provide direct PDF downloads. Use the DOI/URL to reach publisher full text."
        )

    def read_paper(self, paper_id: str, save_path: str = "./downloads") -> str:
        return (
            "PubMed papers cannot be read directly through this tool. "
            "Only metadata and abstracts are available via the PubMed API."
        )
