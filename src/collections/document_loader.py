"""
document_loader.py
==================
Cargador agnóstico de documentos para la colección RAG de originación.

Diseño clave
------------
- Detecta automáticamente los archivos en la carpeta destino.
- Selecciona el extractor correcto según extensión usando un diccionario
  EXTRACTORS: agregar soporte a un nuevo formato = añadir una entrada.
- El resto del pipeline (chunking, embedding, ChromaDB) no cambia
  independientemente del número o tipo de archivos en la carpeta.

Uso
---
    from document_loader import load_documents_from_folder

    docs = load_documents_from_folder("src/collections/documentos_originacion")
    # Retorna lista de dicts: [{"text": str, "source": str, "page": int}, ...]
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTORES por extensión
# Para agregar un nuevo formato: añadir una entrada aquí y definir la función.
# El resto del código no requiere ningún cambio.
# ──────────────────────────────────────────────────────────────────────────────

def _extract_txt(path: Path) -> list[dict]:
    """Extrae texto de archivos .txt y .md."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if not text:
        return []
    return [{"text": text, "source": path.name, "page": 1}]


def _extract_pdf(path: Path) -> list[dict]:
    """Extrae texto de PDFs, página por página, usando pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber no instalado. Ejecuta: pip install pdfplumber")
        return []

    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                # Limpiar espacios múltiples pero preservar saltos de párrafo
                cleaned = "\n".join(
                    line.strip() for line in text.splitlines() if line.strip()
                )
                pages.append({"text": cleaned, "source": path.name, "page": i})
    return pages


def _extract_docx(path: Path) -> list[dict]:
    """Extrae texto de archivos .docx, incluyendo tablas."""
    try:
        from docx import Document
    except ImportError:
        logger.error("python-docx no instalado. Ejecuta: pip install python-docx")
        return []

    doc = Document(path)
    chunks = []

    # Párrafos normales
    para_text = "\n".join(
        p.text.strip() for p in doc.paragraphs if p.text.strip()
    )

    # Contenido de tablas (celda por celda, separadas por tabuladores)
    table_text_parts = []
    for table in doc.tables:
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                table_text_parts.append(" | ".join(row_cells))
    table_text = "\n".join(table_text_parts)

    full_text = "\n\n".join(filter(None, [para_text, table_text])).strip()
    if full_text:
        chunks.append({"text": full_text, "source": path.name, "page": 1})
    return chunks


def _extract_xlsx(path: Path) -> list[dict]:
    """
    Extrae texto de archivos .xlsx.
    Cada hoja se convierte en un chunk independiente para preservar contexto.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas no instalado. Ejecuta: pip install pandas openpyxl")
        return []

    chunks = []
    xl = pd.ExcelFile(path)

    for sheet_name in xl.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, dtype=str)
            df = df.dropna(how="all").fillna("")

            if df.empty:
                continue

            # Serializar: encabezados + filas como texto legible
            lines = []
            lines.append(f"[Hoja: {sheet_name}]")
            lines.append(" | ".join(str(c) for c in df.columns))
            for _, row in df.iterrows():
                row_str = " | ".join(str(v) for v in row.values if str(v).strip())
                if row_str.strip():
                    lines.append(row_str)

            text = "\n".join(lines).strip()
            if text:
                chunks.append({
                    "text": text,
                    "source": path.name,
                    "page": sheet_name   # "page" = nombre de hoja para xlsx
                })
        except Exception as e:
            logger.warning(f"Error leyendo hoja '{sheet_name}' de {path.name}: {e}")

    return chunks


def _extract_csv(path: Path) -> list[dict]:
    """Extrae texto de archivos .csv."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas no instalado.")
        return []

    df = pd.read_csv(path, dtype=str).fillna("")
    lines = [" | ".join(str(c) for c in df.columns)]
    for _, row in df.iterrows():
        lines.append(" | ".join(str(v) for v in row.values))
    text = "\n".join(lines).strip()
    return [{"text": text, "source": path.name, "page": 1}] if text else []


def _extract_json(path: Path) -> list[dict]:
    """Extrae texto de archivos .json serializando como texto legible."""
    import json
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    return [{"text": text, "source": path.name, "page": 1}] if text else []


# ──────────────────────────────────────────────────────────────────────────────
# REGISTRO CENTRAL DE EXTRACTORES
# Agregar un nuevo formato: añadir la extensión y la función aquí.
# ──────────────────────────────────────────────────────────────────────────────
EXTRACTORS: dict[str, callable] = {
    ".txt":  _extract_txt,
    ".md":   _extract_txt,
    ".pdf":  _extract_pdf,
    ".docx": _extract_docx,
    ".xlsx": _extract_xlsx,
    ".xls":  _extract_xlsx,
    ".csv":  _extract_csv,
    ".json": _extract_json,
}

# Extensiones explícitamente ignoradas (no se emite warning)
IGNORED_EXTENSIONS = {".tmp", ".log", ".gitkeep", ".DS_Store", ".ini"}


# ──────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ──────────────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 800,
                overlap: int = 100) -> list[str]:
    """
    Divide un texto largo en chunks con overlap.
    Intenta respetar saltos de párrafo para no cortar ideas a la mitad.
    """
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
                # Overlap: últimas `overlap` chars del chunk anterior
                current = current[-overlap:] + "\n\n" + para
            else:
                # Párrafo solo ya supera chunk_size: corte duro
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
                current = ""

    if current:
        chunks.append(current)

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# API PÚBLICA
# ──────────────────────────────────────────────────────────────────────────────

def load_documents_from_folder(
    folder_path: str,
    chunk_size: int = 800,
    overlap: int = 100,
    verbose: bool = True,
) -> list[dict]:
    """
    Carga y chunkea todos los documentos soportados en `folder_path`.

    Parámetros
    ----------
    folder_path : str
        Ruta a la carpeta con los documentos de normativa.
    chunk_size : int
        Tamaño máximo de cada chunk en caracteres (default 800).
    overlap : int
        Solapamiento entre chunks consecutivos en caracteres (default 100).
    verbose : bool
        Si True, imprime resumen del proceso.

    Retorna
    -------
    list[dict]
        Lista de chunks, cada uno con:
        - "text"   : str  — contenido del chunk
        - "source" : str  — nombre del archivo de origen
        - "page"   : int|str — página o hoja de origen
        - "chunk_id": str — identificador único del chunk
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Carpeta no encontrada: {folder_path}")

    all_chunks = []
    files_processed = 0
    files_skipped = []

    # Ordenar para reproducibilidad
    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()

        if ext in IGNORED_EXTENSIONS:
            continue

        extractor = EXTRACTORS.get(ext)
        if extractor is None:
            files_skipped.append(file_path.name)
            logger.warning(f"Formato no soportado — omitido: {file_path.name}")
            continue

        try:
            raw_docs = extractor(file_path)
        except Exception as e:
            logger.error(f"Error extrayendo {file_path.name}: {e}")
            continue

        file_chunks = []
        for doc in raw_docs:
            text_chunks = _chunk_text(doc["text"], chunk_size, overlap)
            for i, chunk_text in enumerate(text_chunks):
                chunk_id = f"{file_path.stem}_p{doc['page']}_c{i}"
                file_chunks.append({
                    "text":     chunk_text,
                    "source":   doc["source"],
                    "page":     doc["page"],
                    "chunk_id": chunk_id,
                })

        all_chunks.extend(file_chunks)
        files_processed += 1

        if verbose:
            print(f"  ✓ {file_path.name:<45} → {len(file_chunks):>3} chunk(s)")

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Archivos procesados : {files_processed}")
        print(f"  Chunks generados    : {len(all_chunks)}")
        if files_skipped:
            print(f"  Omitidos (sin extractor): {', '.join(files_skipped)}")
        print(f"{'─'*55}")

    return all_chunks


# ──────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN DIRECTA (smoke test)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "src/collections/documentos_originacion"
    print(f"\nCargando documentos desde: {folder}\n")
    chunks = load_documents_from_folder(folder)
    if chunks:
        print(f"\nEjemplo — primer chunk de '{chunks[0]['source']}':")
        print(f"  chunk_id : {chunks[0]['chunk_id']}")
        print(f"  página   : {chunks[0]['page']}")
        print(f"  texto    :\n{chunks[0]['text'][:300]}...")
