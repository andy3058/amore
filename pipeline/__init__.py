"""
AI 헤어 인플루언서 큐레이션 파이프라인 (RAG 기반)
=================================================

파이프라인 구조:
1. Crawlers: 브랜드/인플루언서 데이터 수집
2. Processors: FIS 측정, 분류
3. RAG Analyzer: LLM Vision 이미지 분석 + ChromaDB 벡터 검색
"""

from .crawlers import BrandCrawler, InfluencerCrawler
from .processors import FISCalculator, InfluencerClassifier

# RAG 분석기 (ChromaDB 없으면 graceful 처리)
try:
    from .rag_analyzer import InfluencerImageAnalyzer, InfluencerRAG, InfluencerAnalysisManager
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    InfluencerImageAnalyzer = None
    InfluencerRAG = None
    InfluencerAnalysisManager = None

__all__ = [
    'BrandCrawler',
    'InfluencerCrawler',
    'FISCalculator',
    'InfluencerClassifier',
    'InfluencerImageAnalyzer',
    'InfluencerRAG',
    'InfluencerAnalysisManager',
    'RAG_AVAILABLE'
]
