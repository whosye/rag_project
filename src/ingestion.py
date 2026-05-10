import json
import re
from pathlib import Path

import chromadb
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from fastembed import TextEmbedding


# Matches lines like: "1 Mojžišova 1:1 Na počátku stvořil Bůh…"
_VERSE_RE = re.compile(r"^(\d+)\s+(\S+)\s+(\d+):(\d+)\s")


def _verse_refs(text: str) -> dict:
    """Extract first/last verse references from a chunk of Bible lines."""
    refs = []
    for line in text.splitlines():
        m = _VERSE_RE.match(line.strip())
        if m:
            refs.append({
                "book_num": int(m.group(1)),
                "book": m.group(2),
                "chapter": int(m.group(3)),
                "verse": int(m.group(4)),
            })
    if not refs:
        return {}
    first, last = refs[0], refs[-1]
    return {
        "book": first["book"],
        "book_num": first["book_num"],
        "chapter": first["chapter"],
        "verse_start": first["verse"],
        "verse_end": last["verse"],
        "source": f"{first['book']} {first['chapter']}:{first['verse']}–{last['verse']}",
    }


def main():
    base = Path(__file__).parent.parent
    source = base / "source" / "bkr.txt"
    output = base / "source" / "chunks.jsonl"
    db_path = base / "db"

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    if not output.exists():
        # 1. Načti dokument přes Docling (u PDF/DOCX by zpracoval layout, OCR atd.)
        doc = DocumentConverter().convert(str(source)).document

        # 2. Rozdělí na tokeny-aware chunky respektující hranice odstavců
        chunks = list(HybridChunker().chunk(doc))

        # 3. Obohať každý chunk o metadat specifická pro biblický text a ulož
        with open(output, "w", encoding="utf-8") as f:
            for chunk in chunks:
                record = {"text": chunk.text, "metadata": _verse_refs(chunk.text)}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Ingestováno {len(chunks)} chunků → {output}")

    chunks = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]

    model = TextEmbedding(MODEL_NAME)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = list(model.embed(texts))
    print(f"Vygenerováno {len(embeddings)} embeddingů, dim={len(embeddings[0])}")

    # Uložení do ChromaDB po dávkách (max 5000 najednou)
    db_path.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name="bible")

    BATCH_SIZE = 5000
    all_metadatas = [chunk["metadata"] if chunk["metadata"] else {"source": "unknown"} for chunk in chunks]
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=[str(i) for i in range(start, min(end, len(chunks)))],
            embeddings=[e.tolist() for e in embeddings[start:end]],
            documents=texts[start:end],
            metadatas=all_metadatas[start:end],
        )
        print(f"  dávka {start}–{min(end, len(chunks))} uložena")
    print(f"Uloženo {len(chunks)} chunků do ChromaDB → {db_path}")


if __name__ == "__main__":
    main()
