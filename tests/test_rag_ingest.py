"""Tests for document loading, chunking, and the Chroma ingest→retrieve path.

Chroma / reportlab tests are skipped automatically when those optional deps are absent.
"""

import os

import pytest

from hr_policy_agent.services import rag


def test_split_text_respects_chunk_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 50 for i in range(10))
    chunks = rag._split_text(text, chunk_size=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)


def test_load_chunks_reads_txt_and_md(tmp_path):
    folder = tmp_path / "TOOL"
    folder.mkdir()
    (folder / "a.txt").write_text("Bereavement leave is three days.", encoding="utf-8")
    (folder / "b.md").write_text("# PL\n\nPersonal leave accrues by service.", encoding="utf-8")
    chunks = rag.load_chunks(str(folder))
    titles = {c.metadata["documentTitle"] for c in chunks}
    assert titles == {"a.txt", "b.md"}


def test_load_pdf_and_docx(tmp_path):
    reportlab = pytest.importorskip("reportlab")
    pytest.importorskip("pypdf")
    docx = pytest.importorskip("docx")
    from reportlab.pdfgen import canvas

    folder = tmp_path / "TOOL"
    folder.mkdir()
    pdf_path = folder / "h.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(72, 720, "Voting leave is available on election day.")
    c.save()
    doc = docx.Document()
    doc.add_paragraph("Jury duty leave is provided per policy.")
    doc.save(str(folder / "j.docx"))

    from hr_policy_agent.services.loaders import load_file_text
    assert "Voting leave" in load_file_text(str(pdf_path))

    chunks = rag.load_chunks(str(folder))
    titles = {c.metadata["documentTitle"] for c in chunks}
    assert {"h.pdf", "j.docx"} <= titles


def test_chroma_ingest_and_retrieve(tmp_path):
    pytest.importorskip("langchain_chroma")
    from hr_policy_agent.config import Settings
    from hr_policy_agent.embeddings import get_embeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    settings = Settings(embedding_provider="fake", chroma_dir=str(tmp_path / "chroma"))
    tool = "NON_REP_ENGLISH_DOCUMENT_TOOL_WORKFLOW_V3"
    store = Chroma(collection_name=tool, persist_directory=settings.chroma_dir,
                   embedding_function=get_embeddings(settings))
    store.add_documents([
        Document(page_content="Bereavement leave is three paid days.", metadata={"documentTitle": "hb"}),
        Document(page_content="Parking permits are issued at the security office.", metadata={"documentTitle": "hb"}),
    ])
    retriever = rag._chroma_retriever(tool, settings)
    assert retriever is not None
    docs = retriever.invoke("How many bereavement days?")
    assert any("Bereavement" in d.page_content for d in docs)
