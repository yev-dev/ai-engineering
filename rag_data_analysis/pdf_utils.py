import pymupdf
from pathlib import Path

def get_pdf_text(file_path):
      doc = pymupdf.open(file_path)
      texts = []
      for page in doc:
          temp = page.get_text()
          texts.append(temp.strip())
      doc.close()
      return texts


def render_pdf_pages(file_path, output_dir, zoom=1.5):
    doc = pymupdf.open(file_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_paths = []

    try:
        for index, page in enumerate(doc, start=1):
            image_path = output_path / f"page_{index}.png"
            if not image_path.exists():
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                image_path.write_bytes(pixmap.tobytes("png"))
            image_paths.append(image_path)
    finally:
        doc.close()

    return image_paths