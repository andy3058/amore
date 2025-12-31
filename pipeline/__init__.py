"""
AI 헤어 인플루언서 큐레이션 파이프라인 (학술적 알고리즘 기반)
=============================================================

학술적 기반:
- FIS: Benford's Law (Golbeck, 2015) + Chi-squared Test + Modified Z-score
- 분류: TF-IDF + Cosine Similarity (Salton & McGill, 1983)
- 추천: RRF (Cormack et al., 2009) + Temperature Scaling
- 평가: NDCG (Järvelin & Kekäläinen, 2002) + Diversity

파이프라인 구조:
1. Crawlers: 브랜드/인플루언서 데이터 수집
2. Processors: FIS 측정 (Benford + 통계 검정), TF-IDF 분류, 추천 평가
3. RAG Analyzer: LLM Vision 이미지 분석 + ChromaDB 벡터 검색 + Hybrid Scoring
"""

from .crawlers import BrandCrawler, InfluencerCrawler
from .processors import FISCalculator, InfluencerClassifier, RecommendationEvaluator

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
    'RecommendationEvaluator',
    'InfluencerImageAnalyzer',
    'InfluencerRAG',
    'InfluencerAnalysisManager',
    'RAG_AVAILABLE'
]
