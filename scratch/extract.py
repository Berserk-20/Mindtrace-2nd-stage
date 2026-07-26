import PyPDF2

def extract_pdf_info():
    with open('SAMPLE_Group ID 4 Final Report (Black Book).pdf', 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        # Extract first 15 pages to cover Title, Certificate, Declaration, Acknowledgement, Table of Contents
        num_pages = min(15, len(reader.pages))
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            print(f"--- PAGE {i+1} ---")
            print(text)

if __name__ == "__main__":
    extract_pdf_info()
