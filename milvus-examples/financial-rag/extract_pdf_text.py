import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        str: Extracted text from the PDF
    """
    # Open the PDF file in binary mode
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Get the number of pages in the PDF
        num_pages = len(pdf_reader.pages)
        
        # Extract text from each page
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
        
        return text

if __name__ == "__main__":
    # Path to the PDF file
    pdf_path = "./2024-03-26.pdf"
    
    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Print the first 500 characters of the extracted text
    print("Extracted text (first 500 characters):")
    print(extracted_text[:500])
    
    # Save the extracted text to a file
    output_path = "extracted_text.txt"
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(extracted_text)
    
    print(f"\nFull text saved to {output_path}")