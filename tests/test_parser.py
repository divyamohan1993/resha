import pytest
from unittest.mock import MagicMock, patch
from src.app.services.parser import DocumentParser

@pytest.fixture
def parser():
    return DocumentParser()

def test_parse_txt(parser):
    # Mocking logic for text file
    # Direct access to _parse or simpler execution path
    # Since extract_text takes UploadFile, we need to mock it.
    pass

@pytest.mark.asyncio
async def test_extract_text_txt(parser):
    file_mock = MagicMock()
    file_mock.filename = "test.txt"
    file_mock.read.return_value = b"Hello world"
    
    result = await parser.extract_text(file_mock)
    assert result == "Hello world"

@pytest.mark.asyncio
async def test_extract_text_pdf(parser):
    file_mock = MagicMock()
    file_mock.filename = "test.pdf"
    file_mock.read.return_value = b"%PDF-1.4..."
    
    with patch("src.app.services.parser.PdfReader") as MockReader:
        mock_pdf = MockReader.return_value
        page_mock = MagicMock()
        page_mock.extract_text.return_value = "PDF Content"
        mock_pdf.pages = [page_mock]
        
        result = await parser.extract_text(file_mock)
        assert result == "PDF Content"

@pytest.mark.asyncio
async def test_extract_text_docx(parser):
    file_mock = MagicMock()
    file_mock.filename = "test.docx"
    file_mock.read.return_value = b"PK..."
    
    with patch("src.app.services.parser.Document") as MockDoc:
        mock_doc = MockDoc.return_value
        para_mock = MagicMock()
        para_mock.text = "Docx Content"
        mock_doc.paragraphs = [para_mock]
        
        result = await parser.extract_text(file_mock)
        assert result == "Docx Content"

@pytest.mark.asyncio
async def test_unsupported_format(parser):
    file_mock = MagicMock()
    file_mock.filename = "test.exe"
    file_mock.read.return_value = b"binary"
    
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        await parser.extract_text(file_mock)
