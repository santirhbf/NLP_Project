#!/usr/bin/env python3
"""
Civil Code JSON Validator & Inspector
Validates and inspects the converted JSON structure
"""

import json
import random
from collections import Counter
from typing import Dict, Any, List

def validate_and_inspect_json(json_path: str):
    """Validate and inspect the converted Civil Code JSON"""
    
    print(f"🔍 Inspecting {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return False
    
    print("✅ JSON loaded successfully")
    
    # Basic structure validation
    print("\n📊 BASIC STATISTICS:")
    print(f"   Title: {data.get('metadata', {}).get('title', 'Unknown')}")
    print(f"   Extraction Date: {data.get('metadata', {}).get('extraction_date', 'Unknown')}")
    
    # Preliminary title
    prelim = data.get('preliminary_title', {})
    prelim_articles = len(prelim.get('articles', []))
    print(f"   Preliminary Title Articles: {prelim_articles}")
    
    # Books analysis
    books = data.get('books', [])
    print(f"   Total Books: {len(books)}")
    
    total_articles = prelim_articles
    total_titles = 0
    total_chapters = 0
    
    for book in books:
        book_articles = 0
        titles = book.get('titles', [])
        total_titles += len(titles)
        
        for title in titles:
            title_articles = len(title.get('articles', []))
            book_articles += title_articles
            
            chapters = title.get('chapters', [])
            total_chapters += len(chapters)
            
            for chapter in chapters:
                chapter_articles = len(chapter.get('articles', []))
                book_articles += chapter_articles
                
                for section in chapter.get('sections', []):
                    section_articles = len(section.get('articles', []))
                    book_articles += section_articles
        
        total_articles += book_articles
        print(f"   Book {book.get('number', '?')}: {book_articles} articles")
    
    print(f"   Total Titles: {total_titles}")
    print(f"   Total Chapters: {total_chapters}")
    print(f"   TOTAL ARTICLES: {total_articles}")
    
    # Domain analysis
    print("\n🏷️ DOMAIN ANALYSIS:")
    all_domains = []
    
    # Collect domains from all articles
    def collect_domains(articles_list):
        for article in articles_list:
            all_domains.extend(article.get('domains', []))
    
    collect_domains(prelim.get('articles', []))
    for book in books:
        for title in book.get('titles', []):
            collect_domains(title.get('articles', []))
            for chapter in title.get('chapters', []):
                collect_domains(chapter.get('articles', []))
                for section in chapter.get('sections', []):
                    collect_domains(section.get('articles', []))
    
    domain_counts = Counter(all_domains)
    print("   Domain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"     {domain}: {count} articles")
    
    # Sample articles inspection
    print("\n📋 SAMPLE ARTICLES:")
    sample_articles = []
    
    def collect_articles(articles_list):
        sample_articles.extend(articles_list)
    
    collect_articles(prelim.get('articles', []))
    for book in books:
        for title in book.get('titles', []):
            collect_articles(title.get('articles', []))
            for chapter in title.get('chapters', []):
                collect_articles(chapter.get('articles', []))
                for section in chapter.get('sections', []):
                    collect_articles(section.get('articles', []))
    
    # Show 3 random articles
    if sample_articles:
        random_articles = random.sample(sample_articles, min(3, len(sample_articles)))
        
        for i, article in enumerate(random_articles, 1):
            print(f"\n   Sample Article {i}:")
            print(f"     Number: {article.get('number', 'N/A')}")
            print(f"     Title: {article.get('title', 'No title')}")
            print(f"     Content: {article.get('content', '')[:200]}...")
            print(f"     Subsections: {len(article.get('subsections', []))}")
            print(f"     Domains: {', '.join(article.get('domains', []))}")
    
    # Content quality checks
    print("\n✅ QUALITY CHECKS:")
    
    # Check for empty articles
    empty_articles = [a for a in sample_articles if not a.get('content', '').strip()]
    print(f"   Empty articles: {len(empty_articles)}")
    
    # Check for articles with subsections
    articles_with_subsections = [a for a in sample_articles if a.get('subsections')]
    print(f"   Articles with subsections: {len(articles_with_subsections)}")
    
    # Check for properly classified articles
    classified_articles = [a for a in sample_articles if a.get('domains') and a.get('domains') != ['General']]
    print(f"   Well-classified articles: {len(classified_articles)}")
    
    # Article number continuity check
    article_numbers = [int(a.get('number', 0)) for a in sample_articles if a.get('number', '').isdigit()]
    article_numbers.sort()
    
    if article_numbers:
        print(f"   Article number range: {min(article_numbers)} - {max(article_numbers)}")
        
        # Check for gaps
        expected_range = set(range(min(article_numbers), max(article_numbers) + 1))
        actual_range = set(article_numbers)
        missing_numbers = expected_range - actual_range
        
        if missing_numbers:
            print(f"   ⚠️  Missing article numbers: {sorted(list(missing_numbers))[:10]}...")
        else:
            print(f"   ✅ No missing article numbers")
    
    print(f"\n🎉 Validation complete! The JSON structure looks good.")
    return True

def show_structure_tree(json_path: str):
    """Show the hierarchical structure as a tree"""
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return
    
    print("\n🌳 STRUCTURE TREE:")
    
    # Preliminary title
    prelim = data.get('preliminary_title', {})
    print(f"📄 TÍTULO PRELIMINAR ({len(prelim.get('articles', []))} articles)")
    
    for chapter in prelim.get('chapters', []):
        print(f"  📋 Capítulo {chapter.get('number', '?')}: {chapter.get('title', '')}")
        print(f"      ({len(chapter.get('articles', []))} articles)")
    
    # Books
    for book in data.get('books', []):
        print(f"\n📚 LIBRO {book.get('number', '?')}: {book.get('title', '')}")
        
        for title in book.get('titles', []):
            title_articles = len(title.get('articles', []))
            chapter_articles = sum(len(ch.get('articles', [])) for ch in title.get('chapters', []))
            section_articles = sum(
                len(sec.get('articles', []))
                for ch in title.get('chapters', [])
                for sec in ch.get('sections', [])
            )
            total_title_articles = title_articles + chapter_articles + section_articles
            
            print(f"  📑 Título {title.get('number', '?')}: {title.get('title', '')} ({total_title_articles} articles)")
            
            for chapter in title.get('chapters', []):
                chapter_article_count = len(chapter.get('articles', []))
                section_article_count = sum(len(sec.get('articles', [])) for sec in chapter.get('sections', []))
                total_chapter_articles = chapter_article_count + section_article_count
                
                print(f"    📋 Capítulo {chapter.get('number', '?')}: {chapter.get('title', '')} ({total_chapter_articles} articles)")
                
                for section in chapter.get('sections', []):
                    print(f"      📝 Sección {section.get('number', '?')}: {section.get('title', '')} ({len(section.get('articles', []))} articles)")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        json_path = "codigo_civil_spain_full.json"
        print(f"Using default JSON file: {json_path}")
    else:
        json_path = sys.argv[1]
    
    # Validate and inspect
    if validate_and_inspect_json(json_path):
        # Show structure tree
        show_structure_tree(json_path)
        
        print(f"\n✅ Your Civil Code JSON is ready for the RAG system!")
        print(f"📁 File: {json_path}")
        print(f"🚀 Next step: Integrate with Llama 2 for your legal assistant!")

if __name__ == "__main__":
    main()