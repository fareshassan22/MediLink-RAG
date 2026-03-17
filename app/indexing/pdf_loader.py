from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            pages.append({
                "text": text,
                "page": i + 1
            })

    return pages