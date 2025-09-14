"""
Page number extraction utilities for document processing.
Provides functionality to identify page boundaries and extract page numbers from text.
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class PageInfo:
    """Information about a page in the document."""
    page_number: int
    start_char: int
    end_char: int
    content: str
    page_break_markers: List[str]


@dataclass
class TextLocation:
    """Enhanced text location with page information."""
    start_char: int
    end_char: int
    page_number: int
    page_start_char: int
    page_end_char: int
    relative_position: float  # Position within the page (0.0 to 1.0)


class PageNumberExtractor:
    """Extracts page numbers and boundaries from document text."""
    
    def __init__(self):
        # Common page break patterns
        self.page_break_patterns = [
            r'\f',  # Form feed character
            r'Page\s+\d+',  # "Page 1", "Page 2", etc.
            r'- \d+ -',  # "- 1 -", "- 2 -", etc.
            r'\n\s*\d+\s*\n',  # Standalone page numbers
            r'(?:\n\s*){3,}',  # Multiple blank lines
            r'\n\s*-{3,}\s*\n',  # Horizontal lines
            r'\n\s*={3,}\s*\n',  # Equals lines
            r'\bpage\s+\d+\b',  # "page 1", "page 2" (case insensitive)
            r'\[\s*\d+\s*\]',  # "[1]", "[2]", etc.
        ]
        
        # Patterns for explicit page numbers
        self.page_number_patterns = [
            r'Page\s+(\d+)',
            r'- (\d+) -',
            r'\bpage\s+(\d+)\b',
            r'\[(\d+)\]',
            r'^(\d+)$',  # Standalone number on a line
        ]
    
    def extract_pages(self, text: str) -> List[PageInfo]:
        """Extract page boundaries and information from text."""
        pages = []
        
        # Try to find explicit page breaks
        page_breaks = self._find_page_breaks(text)
        
        if not page_breaks:
            # If no page breaks found, estimate based on text length
            page_breaks = self._estimate_page_breaks(text)
        
        # Create page info objects
        for i, (start_pos, end_pos, markers) in enumerate(page_breaks):
            page_content = text[start_pos:end_pos]
            page_info = PageInfo(
                page_number=i + 1,
                start_char=start_pos,
                end_char=end_pos,
                content=page_content,
                page_break_markers=markers
            )
            pages.append(page_info)
        
        return pages
    
    def _find_page_breaks(self, text: str) -> List[Tuple[int, int, List[str]]]:
        """Find page breaks using various patterns."""
        breaks = []
        last_pos = 0
        
        # Find all potential page break positions
        break_positions = []
        
        for pattern in self.page_break_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                break_positions.append((match.start(), match.end(), pattern))
        
        # Sort by position
        break_positions.sort(key=lambda x: x[0])
        
        # Remove duplicates and overlapping matches
        filtered_breaks = []
        last_end = -1
        
        for start, end, pattern in break_positions:
            if start > last_end:
                filtered_breaks.append((start, end, pattern))
                last_end = end
        
        # Create page segments
        if filtered_breaks:
            current_pos = 0
            page_markers = []
            
            for break_start, break_end, pattern in filtered_breaks:
                if break_start > current_pos:
                    # Add page from current_pos to break_start
                    breaks.append((current_pos, break_start, page_markers.copy()))
                    page_markers = [pattern]
                    current_pos = break_end
                else:
                    page_markers.append(pattern)
            
            # Add final page
            if current_pos < len(text):
                breaks.append((current_pos, len(text), page_markers))
        
        return breaks
    
    def _estimate_page_breaks(self, text: str, chars_per_page: int = 3000) -> List[Tuple[int, int, List[str]]]:
        """Estimate page breaks based on text length when no explicit breaks found."""
        breaks = []
        text_length = len(text)
        
        if text_length <= chars_per_page:
            # Single page
            return [(0, text_length, ['estimated_single_page'])]
        
        # Split into estimated pages
        current_pos = 0
        page_num = 1
        
        while current_pos < text_length:
            # Find a good break point near the estimated page boundary
            ideal_end = min(current_pos + chars_per_page, text_length)
            
            # Look for natural break points (paragraphs, sections)
            break_candidates = []
            search_window = min(200, chars_per_page // 10)  # 10% window
            
            search_start = max(ideal_end - search_window, current_pos)
            search_end = min(ideal_end + search_window, text_length)
            search_text = text[search_start:search_end]
            
            # Look for paragraph breaks
            for match in re.finditer(r'\n\s*\n', search_text):
                abs_pos = search_start + match.start()
                break_candidates.append(abs_pos)
            
            # Look for section headers
            for match in re.finditer(r'\n\s*[A-Z][A-Z\s]{5,}\n', search_text):
                abs_pos = search_start + match.start()
                break_candidates.append(abs_pos)
            
            # Choose best break point
            if break_candidates:
                # Choose the break closest to ideal position
                best_break = min(break_candidates, key=lambda x: abs(x - ideal_end))
                actual_end = best_break
            else:
                actual_end = ideal_end
            
            breaks.append((current_pos, actual_end, [f'estimated_page_{page_num}']))
            current_pos = actual_end
            page_num += 1
        
        return breaks
    
    def get_text_location(self, text: str, start_pos: int, end_pos: int, 
                         pages: Optional[List[PageInfo]] = None) -> TextLocation:
        """Get enhanced location information including page number."""
        if pages is None:
            pages = self.extract_pages(text)
        
        # Find which page contains this text
        page_number = 1
        page_start_char = 0
        page_end_char = len(text)
        
        for page in pages:
            if page.start_char <= start_pos < page.end_char:
                page_number = page.page_number
                page_start_char = page.start_char
                page_end_char = page.end_char
                break
        
        # Calculate relative position within the page
        page_length = page_end_char - page_start_char
        relative_start = start_pos - page_start_char
        relative_position = relative_start / page_length if page_length > 0 else 0.0
        
        return TextLocation(
            start_char=start_pos,
            end_char=end_pos,
            page_number=page_number,
            page_start_char=page_start_char,
            page_end_char=page_end_char,
            relative_position=relative_position
        )
    
    def extract_page_numbers_from_text(self, text: str) -> List[Tuple[int, int, int]]:
        """Extract explicit page numbers and their positions."""
        page_numbers = []
        
        for pattern in self.page_number_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                try:
                    page_num = int(match.group(1))
                    page_numbers.append((match.start(), match.end(), page_num))
                except (IndexError, ValueError):
                    continue
        
        return sorted(page_numbers, key=lambda x: x[0])
    
    def get_page_context(self, text: str, position: int, context_chars: int = 200) -> str:
        """Get surrounding context for a position in the text."""
        start = max(0, position - context_chars)
        end = min(len(text), position + context_chars)
        context = text[start:end]
        
        # Mark the position
        mark_pos = position - start
        if 0 <= mark_pos < len(context):
            context = context[:mark_pos] + ">>>" + context[mark_pos:mark_pos+1] + "<<<" + context[mark_pos+1:]
        
        return context


# Global instance for easy access
page_extractor = PageNumberExtractor()


def get_page_info(text: str, start_pos: int, end_pos: int = None) -> Dict[str, Any]:
    """Convenience function to get page information for a text position."""
    if end_pos is None:
        end_pos = start_pos
    
    pages = page_extractor.extract_pages(text)
    location = page_extractor.get_text_location(text, start_pos, end_pos, pages)
    
    return {
        'page_number': location.page_number,
        'start_char': location.start_char,
        'end_char': location.end_char,
        'page_start_char': location.page_start_char,
        'page_end_char': location.page_end_char,
        'relative_position': location.relative_position,
        'total_pages': len(pages)
    }


def extract_text_with_pages(text: str, start_pos: int, end_pos: int, 
                           context_chars: int = 100) -> Dict[str, Any]:
    """Extract text segment with page information and context."""
    pages = page_extractor.extract_pages(text)
    location = page_extractor.get_text_location(text, start_pos, end_pos, pages)
    
    # Get the actual text segment
    extracted_text = text[start_pos:end_pos]
    
    # Get surrounding context
    context_start = max(0, start_pos - context_chars)
    context_end = min(len(text), end_pos + context_chars)
    context = text[context_start:context_end]
    
    return {
        'extracted_text': extracted_text,
        'context': context,
        'page_number': location.page_number,
        'total_pages': len(pages),
        'position_in_page': location.relative_position,
        'page_boundaries': {
            'start': location.page_start_char,
            'end': location.page_end_char
        },
        'absolute_position': {
            'start': start_pos,
            'end': end_pos
        }
    }
