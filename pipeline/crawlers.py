"""
크롤러 모듈 - 브랜드 및 인플루언서 데이터 수집
==============================================

파이프라인 구조:
    Crawlers (수집) → Processors (분석/분류) → RAG Analyzer (벡터 검색)

모듈 구성:
1. BrandCrawler
   - 아모레퍼시픽 헤어 브랜드 JSON 데이터 관리
   - 수동 입력된 브랜드 정보 로드/저장/검증

2. InfluencerCrawler
   - Instagram Graph API 기반 인플루언서 데이터 수집
   - 해시태그 검색 → 프로필/게시물 수집 → 이미지 URL 수집
   - 수집만 담당 (Expert/Trendsetter 분류는 Processor에서 처리)

지원 브랜드 (6개):
- 려, 미쟝센, 라보에이치, 아윤채, 아모스 프로페셔널, 롱테이크
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# 브랜드 데이터 스키마 (수동 입력용)
BRAND_SCHEMA = {
    "brand_name": "한글 브랜드명",
    "brand_name_en": "영문 브랜드명",
    "slogan": "핵심 슬로건",
    "tagline": "짧은 태그라인",
    "core_values": ["핵심가치1", "핵심가치2"],
    "brand_philosophy": "브랜드 철학",
    "target_keywords": ["타겟키워드1", "타겟키워드2"],
    "aesthetic_style": "Natural/Trendy/Luxury/Classic/Minimal 중 하나",
    "product_categories": ["제품카테고리1", "제품카테고리2"],
    "price_tier": "Premium/Mid-range/Professional/Mass 중 하나",
    "age_target": "타겟 연령층",
    "category": "Hair Care",
    "detail_url": "브랜드 상세 페이지 URL"
}


class BrandCrawler:
    """
    아모레퍼시픽 헤어 브랜드 데이터 관리

    브랜드 데이터는 수동으로 JSON 파일에 입력합니다.
    - 아모레퍼시픽 공식 사이트 조사 후 담당자가 직접 입력
    - data/amore_brands.json 파일에 저장

    이 클래스는 저장된 JSON 데이터를 로드/관리하는 역할만 수행합니다.

    JSON 스키마:
    {
        "brands": {
            "브랜드명": {
                "brand_name": "한글명",
                "brand_name_en": "영문명",
                "slogan": "슬로건",
                "tagline": "태그라인",
                "core_values": ["가치1", "가치2"],
                "brand_philosophy": "브랜드 철학",
                "target_keywords": ["키워드1", "키워드2"],
                "aesthetic_style": "Natural/Trendy/Luxury/Classic/Minimal",
                "product_categories": ["카테고리1", "카테고리2"],
                "price_tier": "Premium/Mid-range/Professional/Mass",
                "age_target": "20-30대",
                "category": "Hair Care",
                "detail_url": "https://..."
            }
        },
        "hair_brands": ["브랜드명1", "브랜드명2"],
        "metadata": {
            "source": "manual_input",
            "last_updated": "2025-01-01T00:00:00",
            "total_brands": 5
        }
    }
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent / "data"

    def load_brands(self) -> Dict:
        """
        저장된 브랜드 데이터 로드

        Returns:
            브랜드 데이터 딕셔너리 (없으면 빈 구조 반환)
        """
        path = self.data_dir / "amore_brands.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"brands": {}, "hair_brands": []}

    def save_brands(self, data: Dict) -> bool:
        """
        브랜드 데이터 저장

        Args:
            data: 브랜드 데이터 딕셔너리

        Returns:
            저장 성공 여부
        """
        try:
            output_path = self.data_dir / "amore_brands.json"
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # 메타데이터 업데이트
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["source"] = "manual_input"
            data["metadata"]["last_updated"] = datetime.now().isoformat()
            data["metadata"]["total_brands"] = len(data.get("brands", {}))

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"브랜드 데이터 저장 완료: {output_path}")
            return True

        except Exception as e:
            logger.error(f"브랜드 데이터 저장 실패: {e}")
            return False

    def add_brand(self, brand_data: Dict) -> bool:
        """
        새 브랜드 추가

        Args:
            brand_data: 브랜드 정보 (BRAND_SCHEMA 참고)

        Returns:
            추가 성공 여부
        """
        try:
            data = self.load_brands()
            brand_name = brand_data.get("brand_name")

            if not brand_name:
                logger.error("브랜드명이 필요합니다")
                return False

            data["brands"][brand_name] = brand_data
            if brand_name not in data.get("hair_brands", []):
                data.setdefault("hair_brands", []).append(brand_name)

            return self.save_brands(data)

        except Exception as e:
            logger.error(f"브랜드 추가 실패: {e}")
            return False

    def get_brand(self, brand_name: str) -> Optional[Dict]:
        """
        특정 브랜드 정보 조회

        Args:
            brand_name: 브랜드명

        Returns:
            브랜드 정보 딕셔너리 또는 None
        """
        data = self.load_brands()
        return data.get("brands", {}).get(brand_name)

    def list_brands(self) -> List[str]:
        """
        등록된 브랜드 목록 반환

        Returns:
            브랜드명 리스트
        """
        data = self.load_brands()
        return data.get("hair_brands", [])

    def validate_brand(self, brand_data: Dict) -> Tuple[bool, List[str]]:
        """
        브랜드 데이터 유효성 검사

        Args:
            brand_data: 검사할 브랜드 데이터

        Returns:
            (유효 여부, 누락된 필드 리스트)
        """
        required_fields = ["brand_name", "brand_name_en", "aesthetic_style", "price_tier"]
        missing = [f for f in required_fields if not brand_data.get(f)]
        return len(missing) == 0, missing

    def get_schema(self) -> Dict:
        """
        브랜드 데이터 스키마 반환 (입력 가이드용)

        Returns:
            스키마 딕셔너리
        """
        return BRAND_SCHEMA.copy()


# ============================================================
# Instagram 인플루언서 크롤러
# ============================================================


class InfluencerCrawler:
    """
    Instagram Graph API 기반 인플루언서 크롤러

    역할: 데이터 수집만 담당 (Expert/Trendsetter 구분 없이)
    - 해시태그 기반 인플루언서 검색
    - Business Discovery API로 프로필/게시물 수집
    - 이미지 URL 수집 (분석은 Processor에서)

    Note:
    - Expert/Trendsetter 분류는 Processor에서 수행
    - 분석 전략 결정도 Processor에서 수행
    """

    # 검색 해시태그 (전문가 + 트렌드세터 통합)
    SEARCH_HASHTAGS = [
        # 헤어 전문가 관련
        "미용사", "헤어디자이너", "살롱", "헤어시술",
        "염색전문", "펌전문", "헤어클리닉", "두피케어",
        # 패션/라이프스타일 관련
        "ootd", "dailylook", "패션", "데일리룩",
        "코디", "스타일링", "패션스타그램", "뷰티그램"
    ]

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent / "data"
        self.access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
        self.business_account_id = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID")
        self.base_url = "https://graph.facebook.com/v18.0"

    def crawl(self, limit_per_hashtag: int = 20) -> Dict:
        """
        인플루언서 크롤링 (구분 없이 수집)

        Args:
            limit_per_hashtag: 해시태그당 수집할 최대 인플루언서 수

        Returns:
            {
                "influencers": [...],  # raw 인플루언서 데이터
                "metadata": {...}
            }
        """
        if not self.access_token or not self.business_account_id:
            logger.warning("Instagram API 토큰 미설정. 샘플 데이터 로드.")
            return self._load_raw_sample_data()

        all_influencers = []
        seen_usernames = set()

        for hashtag in self.SEARCH_HASHTAGS:
            try:
                # 1. 해시태그 ID 검색
                hashtag_url = f"{self.base_url}/ig_hashtag_search"
                params = {
                    "user_id": self.business_account_id,
                    "q": hashtag,
                    "access_token": self.access_token
                }
                response = requests.get(hashtag_url, params=params, timeout=10)
                response.raise_for_status()
                hashtag_data = response.json()

                if not hashtag_data.get("data"):
                    continue

                hashtag_id = hashtag_data["data"][0]["id"]

                # 2. 상위 미디어 검색
                media_url = f"{self.base_url}/{hashtag_id}/top_media"
                params = {
                    "user_id": self.business_account_id,
                    "fields": "id,caption,media_type,permalink,timestamp,"
                              "like_count,comments_count,owner{username}",
                    "limit": limit_per_hashtag,
                    "access_token": self.access_token
                }
                response = requests.get(media_url, params=params, timeout=10)
                response.raise_for_status()
                media_data = response.json()

                # 3. 각 미디어에서 사용자 추출
                for media in media_data.get("data", []):
                    owner = media.get("owner", {})
                    username = owner.get("username")

                    if username and username not in seen_usernames:
                        seen_usernames.add(username)
                        details = self.get_influencer_details(username)
                        if details:
                            all_influencers.append(details)

            except Exception as e:
                logger.warning(f"해시태그 '{hashtag}' 검색 실패: {e}")
                continue

        result = {
            "influencers": all_influencers,
            "metadata": {
                "search_hashtags": self.SEARCH_HASHTAGS,
                "total_count": len(all_influencers),
                "crawled_at": datetime.now().isoformat(),
                "status": "raw"  # 아직 분류/분석 안됨
            }
        }

        # raw 데이터 저장
        self._save_raw_data(result)
        return result

    def get_influencer_details(self, username: str) -> Optional[Dict]:
        """
        특정 인플루언서의 상세 정보 수집

        Business Discovery API를 통해 수집 가능한 필드:
        - 기본 정보: username, name, biography, followers_count, media_count
        - 미디어: id, caption, like_count, comments_count, media_type, timestamp, media_url, permalink
        """
        if not self.access_token:
            return None

        try:
            url = f"{self.base_url}/{self.business_account_id}"
            params = {
                "fields": f"business_discovery.username({username})"
                          "{username,name,biography,followers_count,media_count,"
                          "media.limit(12){id,caption,like_count,comments_count,"
                          "media_type,timestamp,media_url,permalink}}",
                "access_token": self.access_token
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            discovery = data.get("business_discovery", {})
            if not discovery:
                return None

            return self._normalize_influencer_data(discovery)

        except Exception as e:
            logger.error(f"인플루언서 상세 정보 수집 실패 ({username}): {e}")
            return None

    def _normalize_influencer_data(self, raw_data: Dict) -> Dict:
        """Instagram API 응답을 표준 스키마로 정규화 (raw 데이터)"""
        username = raw_data.get("username", "")

        recent_posts = []
        media_list = raw_data.get("media", {}).get("data", [])
        timestamps = []

        for media in media_list:
            post = {
                "caption": media.get("caption", ""),
                "likes": media.get("like_count", 0),
                "comments": media.get("comments_count", 0),
                "media_type": media.get("media_type", ""),
                "timestamp": media.get("timestamp", ""),
                "media_url": media.get("media_url", ""),
                "permalink": media.get("permalink", "")
            }
            recent_posts.append(post)

            if media.get("timestamp"):
                timestamps.append(media["timestamp"])

        avg_interval = self._calculate_upload_interval(timestamps)

        return {
            "username": username,
            "followers": raw_data.get("followers_count", 0),
            "bio": raw_data.get("biography", ""),
            "media_count": raw_data.get("media_count", 0),
            "recent_posts": recent_posts,
            "audience_countries": {},
            "avg_upload_interval_days": avg_interval
        }

    def _calculate_upload_interval(self, timestamps: List[str]) -> float:
        """평균 업로드 간격 계산"""
        if len(timestamps) < 2:
            return 0.0

        try:
            dates = sorted([
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                for ts in timestamps
            ], reverse=True)

            intervals = []
            for i in range(len(dates) - 1):
                diff = (dates[i] - dates[i+1]).days
                intervals.append(diff)

            return sum(intervals) / len(intervals) if intervals else 0.0
        except Exception:
            return 0.0

    def _save_raw_data(self, data: Dict) -> None:
        """raw 크롤링 데이터 저장"""
        output_path = self.data_dir / "influencers_raw.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Raw 인플루언서 데이터 저장: {output_path}")

    def _load_raw_sample_data(self) -> Dict:
        """샘플 데이터 로드 (API 없을 때)"""
        path = self.data_dir / "influencers_raw.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # influencers_data.json에서 raw 형식으로 변환
        data_path = self.data_dir / "influencers_data.json"
        if data_path.exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 분석 결과 제거하고 raw 데이터만 반환
                raw_influencers = []
                for inf in data.get("influencers", []):
                    raw_inf = {k: v for k, v in inf.items()
                              if k not in ["influencer_type", "analysis_strategy",
                                           "text_analysis", "image_analysis"]}
                    raw_influencers.append(raw_inf)
                return {
                    "influencers": raw_influencers,
                    "metadata": {
                        "status": "raw",
                        "source": "sample_data"
                    }
                }

        return {"influencers": [], "metadata": {"status": "empty"}}

    def load_raw_data(self) -> Dict:
        """저장된 raw 크롤링 데이터 로드"""
        return self._load_raw_sample_data()


# 테스트
if __name__ == "__main__":
    # 브랜드 크롤러 테스트
    brand_crawler = BrandCrawler()
    brands = brand_crawler.load_brands()
    print(f"로드된 브랜드: {len(brands.get('brands', {}))}")

    # 인플루언서 크롤러 테스트
    inf_crawler = InfluencerCrawler()

    # 크롤링 (API 없으면 샘플 데이터 로드)
    result = inf_crawler.crawl()
    print(f"크롤링된 인플루언서: {result['metadata']['total_count']}명")
    print(f"상태: {result['metadata']['status']}")

    if result['influencers']:
        sample = result['influencers'][0]
        print(f"\n샘플: @{sample['username']}")
        print(f"  Bio: {sample['bio'][:50]}..." if sample['bio'] else "  Bio: (없음)")
        print(f"  Followers: {sample['followers']:,}")
        print(f"  Posts: {len(sample['recent_posts'])}개")
