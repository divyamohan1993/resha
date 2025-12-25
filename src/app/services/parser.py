import io
from pypdf import PdfReader
from docx import Document
from fastapi import UploadFile, HTTPException
from ..core.logging import get_logger

logger = get_logger(__name__)

class DocumentParser:
    async def extract_text(self, file: UploadFile) -> str:
        filename = file.filename.lower()
        content = await file.read()
        file_obj = io.BytesIO(content)
        
        try:
            if filename.endswith('.pdf'):
                return self._parse_pdf(file_obj)
            elif filename.endswith('.docx'):
                return self._parse_docx(file_obj)
            elif filename.endswith('.txt'):
                return content.decode('utf-8')
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF, DOCX, or TXT.")
        except Exception as e:
            logger.error(f"Error parsing file {filename}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    def _parse_pdf(self, file_obj) -> str:
        reader = PdfReader(file_obj)
        text = []
        for page in reader.pages:
            result = page.extract_text()
            if result:
                text.append(result)
        return "\n".join(text)

    def _parse_docx(self, file_obj) -> str:
        doc = Document(file_obj)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

parser_service = DocumentParser()
