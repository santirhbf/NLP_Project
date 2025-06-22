#!/usr/bin/env python3
"""
Fix Preliminary Title Articles (Articles 1-16)
Extracts and adds the missing preliminary title articles to the JSON
"""

import json
import re
import fitz  # PyMuPDF
import sys

def extract_preliminary_articles(pdf_path: str, json_path: str):
    """Extract preliminary title articles and add them to existing JSON"""
    
    print("🔍 Extracting Título Preliminar articles...")
    
    # Open PDF and extract text around articles 1-16
    doc = fitz.open(pdf_path)
    preliminary_text = ""
    
    # Look for preliminary title in first 50 pages
    for page_num in range(min(50, len(doc))):
        page = doc[page_num]
        text = page.get_text()
        
        # Look for preliminary title or early articles
        if any(indicator in text.upper() for indicator in [
            "TÍTULO PRELIMINAR", 
            "FUENTES DEL DERECHO",
            "Artículo 1.",
            "CAPÍTULO I"
        ]):
            preliminary_text += text + "\n"
            print(f"   Found preliminary content on page {page_num + 1}")
    
    doc.close()
    
    if not preliminary_text:
        print("❌ No preliminary title content found")
        return False
    
    print(f"✅ Extracted {len(preliminary_text)} characters of preliminary content")
    
    # Extract articles 1-16 specifically
    articles = extract_articles_1_to_16(preliminary_text)
    
    if not articles:
        print("❌ No articles 1-16 found")
        return False
    
    print(f"✅ Found {len(articles)} preliminary articles")
    
    # Load existing JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return False
    
    # Create preliminary_title if it doesn't exist
    if 'preliminary_title' not in data:
        data['preliminary_title'] = {
            "title": "TÍTULO PRELIMINAR - De las normas jurídicas, su aplicación y eficacia",
            "articles": [],
            "chapters": []
        }
    
    # Update preliminary title
    data['preliminary_title']['articles'] = articles
    
    # Also extract chapters if found
    chapters = extract_preliminary_chapters(preliminary_text)
    if chapters:
        data['preliminary_title']['chapters'] = chapters
        print(f"✅ Added {len(chapters)} preliminary chapters")
    
    # Save updated JSON
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Updated {json_path} with preliminary articles")
        return True
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
        return False

def extract_articles_1_to_16(text: str) -> list:
    """Extract specifically articles 1-16 from preliminary title"""
    articles = []
    
    # Look for articles numbered 1 through 16
    for article_num in range(1, 17):
        # Multiple patterns to catch different formatting
        patterns = [
            rf'Artículo\s+{article_num}\.?\s*(.*?)(?=Artículo\s+{article_num + 1}|CAPÍTULO|TÍTULO|$)',
            rf'Art\.\s+{article_num}\.?\s*(.*?)(?=Art\.\s+{article_num + 1}|CAPÍTULO|TÍTULO|$)',
            rf'^{article_num}\.?\s+(.*?)(?=^{article_num + 1}\.|CAPÍTULO|TÍTULO|$)'
        ]
        
        article_found = False
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                content = match.group(1).strip()
                if len(content) > 20:  # Ensure it's substantial content
                    article = parse_preliminary_article(article_num, content)
                    if article:
                        articles.append(article)
                        article_found = True
                        print(f"   ✅ Found Article {article_num}")
                        break
            
            if article_found:
                break
        
        if not article_found:
            print(f"   ⚠️  Article {article_num} not found")
    
    return articles

def parse_preliminary_article(number: int, content: str) -> dict:
    """Parse a single preliminary article"""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    if not lines:
        return None
    
    # Clean up content - remove page breaks and artifacts
    cleaned_lines = []
    for line in lines:
        # Skip obvious artifacts
        if re.match(r'^\d+$', line):  # Just a number
            continue
        if len(line) < 3:  # Too short
            continue
        if 'CÓDIGO CIVIL' in line.upper():
            continue
        cleaned_lines.append(line)
    
    if not cleaned_lines:
        return None
    
    # First line might be a title
    title = None
    content_start = 0
    
    if not re.match(r'^\d+\.', cleaned_lines[0]):
        title = cleaned_lines[0]
        content_start = 1
    
    # Parse subsections
    main_content = []
    subsections = []
    
    for line in cleaned_lines[content_start:]:
        # Check for numbered subsections
        if num_match := re.match(r'^(\d+)\.?\s+(.*)', line):
            subsections.append({
                "type": "numbered",
                "number": num_match.group(1),
                "content": num_match.group(2)
            })
        # Check for lettered subsections
        elif letter_match := re.match(r'^([a-z])\)\s+(.*)', line):
            subsections.append({
                "type": "lettered", 
                "letter": letter_match.group(1),
                "content": letter_match.group(2)
            })
        else:
            main_content.append(line)
    
    # Classify domains
    domains = classify_preliminary_article(number, title or "", " ".join(main_content))
    
    return {
        "number": str(number),
        "title": title,
        "content": " ".join(main_content),
        "subsections": subsections,
        "domains": domains
    }

def extract_preliminary_chapters(text: str) -> list:
    """Extract chapters from preliminary title"""
    chapters = []
    
    # Look for chapters in preliminary title
    chapter_pattern = r'CAPÍTULO\s+([IVX]+)\.?\s*(.*?)(?=CAPÍTULO\s+[IVX]+|LIBRO|$)'
    matches = re.finditer(chapter_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        chapter_num = match.group(1)
        chapter_title = match.group(2).split('\n')[0].strip()
        
        # Extract articles in this chapter
        chapter_text = match.group(2)
        chapter_articles = []
        
        # Find articles in this chapter (rough approximation)
        article_matches = re.finditer(r'Artículo\s+(\d+)', chapter_text, re.IGNORECASE)
        article_numbers = [int(m.group(1)) for m in article_matches if 1 <= int(m.group(1)) <= 16]
        
        chapters.append({
            "number": chapter_num,
            "title": chapter_title,
            "articles": [],  # Articles will be linked by number later
            "article_numbers": article_numbers  # Helper for validation
        })
    
    return chapters

def classify_preliminary_article(number: int, title: str, content: str) -> list:
    """Classify preliminary articles into domains"""
    text = f"{title} {content}".lower()
    
    # Specific classification for preliminary articles
    if 1 <= number <= 7:
        return ["Fuentes del Derecho"]
    elif 8 <= number <= 12:
        return ["Aplicación de Normas"]
    elif 13 <= number <= 16:
        return ["Derecho Internacional Privado"]
    else:
        return ["Título Preliminar"]

def main():
    """Main execution"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python fix_preliminary_title.py <pdf_path> <json_path>")
        print("Example: python fix_preliminary_title.py codigo_civil.pdf codigo_civil_spain_full.json")
        return
    
    pdf_path = sys.argv[1]
    json_path = sys.argv[2]
    
    success = extract_preliminary_articles(pdf_path, json_path)
    
    if success:
        print("\n🎉 Preliminary title articles successfully added!")
        print("📋 Run the validator again to confirm:")
        print(f"   python json_validator.py {json_path}")
    else:
        print("\n❌ Failed to extract preliminary articles")

def quick_fix():
    """Quick fix using default file names"""
    return extract_preliminary_articles("codigo_civil.pdf", "codigo_civil_spain_full.json")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🔧 Running quick fix with default file names...")
        if quick_fix():
            print("✅ Quick fix successful!")
        else:
            print("❌ Quick fix failed - try manual mode")
    else:
        main()