"""
설정 모듈 - 제품 카테고리 및 Instagram API 설정
================================================

products.py:
- PRODUCT_CATEGORIES: 아모레퍼시픽 헤어 제품 카테고리 (10개)
- PRODUCT_KEYWORDS: 자연어 추출용 키워드 매핑
- BRAND_PRODUCT_LINES: 브랜드별 대표 제품 라인

instagram.py:
- Instagram Graph API 설정
- Rate Limit, 캐시 설정
- 헤어 관련 해시태그 목록
"""
from .products import PRODUCT_CATEGORIES, PRODUCT_KEYWORDS, BRAND_PRODUCT_LINES
from .instagram import (
    INSTAGRAM_API_VERSION,
    INSTAGRAM_ACCESS_TOKEN,
    INSTAGRAM_BUSINESS_ACCOUNT_ID,
    HAIR_HASHTAGS
)