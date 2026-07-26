import docx

doc = docx.Document(r"c:\Users\sanka\MindTrace\MindTrace_Research_Paper_Final_v10.docx")

for i, section in enumerate(doc.sections):
    hfs = [
        (section.header, f"Section {i} Header"),
        (section.first_page_header, f"Section {i} First Page Header"),
        (section.even_page_header, f"Section {i} Even Page Header"),
        (section.footer, f"Section {i} Footer"),
        (section.first_page_footer, f"Section {i} First Page Footer"),
        (section.even_page_footer, f"Section {i} Even Page Footer")
    ]
    
    for hf, name in hfs:
        for p in hf.paragraphs:
            if p.text.strip():
                print(f"{name}: {p.text}")
