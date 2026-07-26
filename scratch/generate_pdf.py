import re
from fpdf import FPDF
import markdown

with open("diagrams_explanation.md", "r", encoding="utf-8") as f:
    text = f.read()

# Strip markdown formatting slightly for fpdf (since it's basic)
text = text.replace("#### ", "\n\n--- ")
text = text.replace("### ", "\n\n")
text = text.replace("**", "")
text = text.replace("👤", "")
text = text.replace("🖥️", "")
text = text.replace("⚙️", "")
text = text.replace("🗄️", "")

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=11)

for line in text.split('\n'):
    try:
        # Just write lines directly, handling basic encoding issues if any
        pdf.multi_cell(0, 6, txt=line.encode('latin-1', 'replace').decode('latin-1'))
    except Exception as e:
        pdf.multi_cell(0, 6, txt=str(e))

pdf.output("diagrams_explanation.pdf")
print("PDF created successfully.")
