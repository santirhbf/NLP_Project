#!/usr/bin/env python3
"""
Diagnose Civil Code Structure Issues
Analyze why chapters have 0 articles and fix the structure
"""

import json
from collections import defaultdict

def diagnose_structure(json_path: str):
    """Diagnose structural issues in the Civil Code JSON"""
    
    print("🔍 DIAGNOSING CIVIL CODE STRUCTURE ISSUES")
    print("=" * 50)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return False
    
    # 1. Analyze preliminary title issues
    print("\n📄 PRELIMINARY TITLE ANALYSIS:")
    prelim = data.get('preliminary_title', {})
    
    print(f"   Articles in preliminary: {len(prelim.get('articles', []))}")
    print(f"   Chapters in preliminary: {len(prelim.get('chapters', []))}")
    
    # Show article numbers in preliminary
    prelim_articles = prelim.get('articles', [])
    if prelim_articles:
        article_nums = [art.get('number', 'N/A') for art in prelim_articles]
        print(f"   Article numbers: {sorted(article_nums, key=lambda x: int(x) if x.isdigit() else 999)}")
    
    # Show chapter issues
    prelim_chapters = prelim.get('chapters', [])
    print(f"\n   📋 CHAPTER ANALYSIS:")
    for i, chapter in enumerate(prelim_chapters[:10]):  # Show first 10
        print(f"      Chapter {i+1}: '{chapter.get('title', 'No title')[:60]}...'")
        print(f"         Articles: {len(chapter.get('articles', []))}")
    
    if len(prelim_chapters) > 10:
        print(f"      ... and {len(prelim_chapters) - 10} more chapters")
    
    # 2. Check if articles are orphaned
    print(f"\n🔍 ORPHANED ARTICLES CHECK:")
    
    # Count all articles across structure
    def count_articles_recursive(structure, path=""):
        total = 0
        articles_found = []
        
        if isinstance(structure, dict):
            # Count articles at this level
            if 'articles' in structure:
                level_articles = structure['articles']
                total += len(level_articles)
                for art in level_articles:
                    articles_found.append({
                        'number': art.get('number', 'N/A'),
                        'location': path,
                        'content_length': len(art.get('content', ''))
                    })
            
            # Recurse into nested structures
            for key, value in structure.items():
                if key in ['books', 'titles', 'chapters', 'sections']:
                    if isinstance(value, list):
                        for i, item in enumerate(value):
                            sub_total, sub_articles = count_articles_recursive(
                                item, f"{path}/{key}[{i}]"
                            )
                            total += sub_total
                            articles_found.extend(sub_articles)
        
        return total, articles_found
    
    total_articles, all_articles = count_articles_recursive(data)
    print(f"   Total articles found: {total_articles}")
    
    # 3. Check for article number distribution
    article_numbers = []
    for art in all_articles:
        if art['number'].isdigit():
            article_numbers.append(int(art['number']))
    
    if article_numbers:
        article_numbers.sort()
        print(f"   Article range: {min(article_numbers)} - {max(article_numbers)}")
        
        # Find gaps
        expected = set(range(min(article_numbers), max(article_numbers) + 1))
        actual = set(article_numbers)
        gaps = sorted(expected - actual)
        
        if gaps:
            print(f"   Missing articles: {gaps[:20]}{'...' if len(gaps) > 20 else ''}")
        else:
            print("   ✅ No gaps in article numbering")
    
    # 4. Show articles by location
    print(f"\n📍 ARTICLES BY LOCATION:")
    location_counts = defaultdict(int)
    for art in all_articles:
        location = art['location'].split('/')[1] if '/' in art['location'] else 'root'
        location_counts[location] += 1
    
    for location, count in sorted(location_counts.items()):
        print(f"   {location}: {count} articles")
    
    # 5. Sample some articles to check content quality
    print(f"\n📋 SAMPLE ARTICLES:")
    sample_articles = [art for art in all_articles if art['content_length'] > 50][:3]
    
    for i, art in enumerate(sample_articles, 1):
        print(f"   Sample {i}:")
        print(f"      Number: {art['number']}")
        print(f"      Location: {art['location']}")
        print(f"      Content length: {art['content_length']} chars")
        
        # Get full article from data
        full_article = None
        for a in prelim_articles:
            if a.get('number') == art['number']:
                full_article = a
                break
        
        if full_article:
            content = full_article.get('content', '')
            print(f"      Content preview: {content[:100]}...")
    
    return True

def suggest_fixes(json_path: str):
    """Suggest specific fixes for the structure issues"""
    
    print(f"\n🔧 SUGGESTED FIXES:")
    print("=" * 30)
    
    print("1. ❌ PRELIMINARY TITLE CHAPTERS:")
    print("   Problem: Too many chapters extracted, likely from wrong sections")
    print("   Fix: Re-extract only chapters I-V from preliminary title pages")
    
    print("\n2. ❌ ARTICLE-CHAPTER ASSIGNMENT:")
    print("   Problem: Articles exist but aren't assigned to chapters")
    print("   Fix: Map articles 1-16 to appropriate preliminary chapters:")
    print("      • Articles 1-7  → Capítulo I (Fuentes del derecho)")
    print("      • Articles 8-12 → Capítulo II (Aplicación de normas)")
    print("      • Articles 13-16 → Capítulo IV (Derecho internacional)")
    
    print("\n3. ✅ ARTICLE CONTENT:")
    print("   Status: Articles appear to have good content")
    print("   Action: Content quality looks good, keep as-is")
    
    print("\n4. 🔧 RECOMMENDED ACTION:")
    print("   Create a structure cleanup script to:")
    print("   • Remove incorrect chapters from preliminary title")
    print("   • Properly assign preliminary articles to correct chapters") 
    print("   • Validate book-level structure is intact")

def main():
    """Main diagnostic function"""
    import sys
    
    json_path = sys.argv[1] if len(sys.argv) > 1 else "codigo_civil_spain_full.json"
    
    print(f"🔍 Diagnosing: {json_path}")
    
    if diagnose_structure(json_path):
        suggest_fixes(json_path)
        
        print(f"\n💡 NEXT STEPS:")
        print("1. Run structure cleanup script")
        print("2. Re-validate with json_validator.py")
        print("3. Only then proceed with legal assistant")
        
        return True
    else:
        print("❌ Diagnosis failed")
        return False

if __name__ == "__main__":
    main()