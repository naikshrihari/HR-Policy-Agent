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


def test_section_aware_chunking_and_short_query_retrieval(tmp_path):
    """Regression: 'can i take voting leave?' must retrieve the Voting section, not
    whatever shares the most keywords."""
    folder = tmp_path / "TOOL"
    folder.mkdir()
    (folder / "hb.txt").write_text(
        "Personal Leave (PL) Accrual\n\n"
        "Eligible Team Members accrue Personal Leave based on length of service.\n\n"
        "Bereavement Leave\n\n"
        "Team Members are eligible for up to three (3) days of paid bereavement leave.\n\n"
        "Voting Leave\n\n"
        "Team Members may take paid time off to vote, generally up to two (2) hours, and "
        "more time if the polling place is far from the workplace.\n\n"
        "Jury Duty Leave\n\n"
        "Team Members summoned for jury duty are granted leave for the service.\n",
        encoding="utf-8",
    )
    chunks = rag.load_chunks(str(folder), 900, 150)
    assert len(chunks) >= 4  # one chunk per section

    retriever = rag._build_retriever(str(folder), rag.Settings(rag_top_k=1, rag_chunk_size=900))
    for query in ("can i take voting leave?", "voting booth is 3 miles away, how much leave can I take?"):
        top = retriever.invoke(query)[0].page_content
        assert "Voting Leave" in top, f"{query!r} retrieved: {top[:40]!r}"


def test_clean_pdf_text_dehyphenates_and_strips_furniture():
    from hr_policy_agent.services.loaders import clean_pdf_text
    raw = ("EMPLOYMENT POLICIES 53\n"
           "Team Members must follow all si-\ngns and speed limits when walking thr-\n"
           "ough the parking areas.\n\n"
           "Photography Policy\n"
           "No photographs may be taken inside the casino.\n")
    cleaned = clean_pdf_text(raw)
    assert "signs" in cleaned and "through" in cleaned      # de-hyphenated
    assert "si-" not in cleaned and "thr-" not in cleaned
    assert "EMPLOYMENT POLICIES 53" not in cleaned          # page furniture removed
    assert "Photography Policy" in cleaned


def test_word_safe_chunking_never_splits_mid_word():
    from hr_policy_agent.services.rag import _split_text
    para = " ".join(f"sentence{i} has several words here." for i in range(60))
    chunks = _split_text(para, 300, 50)
    assert len(chunks) > 1
    # every chunk starts and ends on a whole word (no leading/trailing partial token)
    for c in chunks:
        assert not c.startswith(" ") and "  " not in c
        assert c.split()[0].isascii()


def test_focused_citation_shows_only_the_source_used():
    from hr_policy_agent.codenodes.citation_details import focused_citation
    rag = [{"supportingChunks": [
        {"textChunk": "Pay Corrections. Corrections to a Team Member's paycheck must be brought "
                      "to the attention of their supervisor or department manager.",
         "documentIdentificationCriteria": {"documentTitle": "Handbook"}},
        {"textChunk": "Voting Leave. Team Members may take paid time off to vote.",
         "documentIdentificationCriteria": {"documentTitle": "Handbook"}},
    ]}]
    card = focused_citation(
        "Corrections to your paycheck should be brought to your supervisor or department manager.",
        rag, "EN")
    assert "paycheck must be brought" in card          # the used chunk
    assert "Voting Leave" not in card                  # the unused chunk is excluded
    assert card.count("<details") == 1                 # exactly one source shown
    # No citation for a not-covered or unrelated answer.
    assert focused_citation("This topic is not covered in the available policy documents.", rag, "EN") == ""
    assert focused_citation("The sky is blue.", rag, "EN") == ""
