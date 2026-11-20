import io

import pytest
from pypdf import PdfReader, PdfWriter

from airlock.pdf import load_pdf
from airlock.sample import build_sample_pdf


def _encrypted(user_password: str) -> bytes:
    data, _ = build_sample_pdf(pages=5)
    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(data)).pages:
        writer.add_page(page)
    writer.encrypt(user_password=user_password, owner_password="owner", algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_pdf_with_empty_user_password_still_opens():
    doc = load_pdf(_encrypted(""), "locked.pdf")
    assert doc.page_count == 5 and not doc.empty_pages


def test_password_protected_pdf_gives_a_clear_error():
    with pytest.raises(ValueError, match="password-protected"):
        load_pdf(_encrypted("hunter2"), "secret.pdf")


def test_non_pdf_gives_a_clear_error():
    with pytest.raises(ValueError, match="doesn't look like a readable PDF"):
        load_pdf(b"this is not a pdf", "notes.txt")
