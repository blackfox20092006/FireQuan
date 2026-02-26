import fitz
import sys

def read_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        out = ""
        for i in range(min(5, doc.page_count)):  # read first 5 pages for summary
            out += f"--- Page {i+1} ---\n"
            out += doc[i].get_text()
        print(out)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        read_pdf(sys.argv[1])
    else:
        print("Usage: python script.py <pdf_file>")
