"""
Text cleaning and chunking utilities for raw document preprocessing.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Strip noise from raw plan/grant text so embeddings aren't polluted."""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[•▪▶►◆●○■□–—]", " ", text)

    # kill numbered prefixes (1. / 2.1 / etc) that appear at line starts
    # careful not to eat years or budget figures mid-sentence
    text = re.sub(r"(?m)^\s*\d{1,2}\.\d{0,2}\s*", " ", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split long text into overlapping word-level chunks.
    Short texts come back as a single-element list.
    """
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break
        start += step

    return chunks


def extract_sections(text: str) -> dict[str, str]:
    """
    Best-effort section splitting based on numbered headers.
    Returns {"full": text} if nothing parseable is found.
    """
    pattern = r"(?m)^(\d{1,2}(?:\.\d{1,2})?\s+[A-Z\u00C0-\u00DC][A-Za-z\u00C0-\u00FF\s&,]+)"
    splits = re.split(pattern, text)

    if len(splits) < 3:
        return {"full": text}

    sections = {}

    # splits alternates: [preamble, header1, content1, header2, content2, ...]
    for i in range(1, len(splits) - 1, 2):
        header = splits[i].strip()
        content = splits[i + 1].strip() if (i + 1) < len(splits) else ""
        sections[header] = content

    preamble = splits[0].strip()
    if preamble:
        sections["_preamble"] = preamble

    return sections


if __name__ == "__main__":
    sample = """
    1. PREMESSA
    Il presente Piano   Strategico delinea la visione
    di sviluppo sostenibile del Comune.

    • Installazione di impianti fotovoltaici
    • Creazione di una comunità energetica
    ▶ Riqualificazione    energetica degli edifici

    2.1 Transizione Energetica
    Il Comune intende raggiungere la neutralità carbonica.
    """

    cleaned = clean_text(sample)
    print("Cleaned:", cleaned[:120], "...")
    print()

    chunks = chunk_text(cleaned, chunk_size=15, overlap=3)
    for i, c in enumerate(chunks):
        print(f"  chunk {i}: {c}")
