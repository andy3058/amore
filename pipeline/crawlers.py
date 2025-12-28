"""
크롤러 모듈 - 브랜드 및 인플루언서 데이터 수집
==============================================

1. BrandCrawler: 아모레퍼시픽 헤어 브랜드 크롤링 + LLM 구조화
2. InfluencerCrawler: Instagram Graph API 기반 인플루언서 데이터 수집
"""

import os
import json
import logging
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

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

# 헤어 브랜드 URL
HAIR_BRANDS_URL = "https://www.apgroup.com/int/ko/brands/brands.html#Hair"
BASE_URL = "https://www.apgroup.com"

# LLM 브랜드 구조화 프롬프트
BRAND_EXTRACTION_PROMPT = """브랜드 정보를 분석하여 구조화된 JSON으로 추출하세요.

브랜드명: {brand_name}
슬로건: {slogan}
설명: {description}
상세페이지 URL: {detail_url}

JSON 형식:
{{
  "brand_name": "한글 브랜드명",
  "brand_name_en": "영문 브랜드명",
  "slogan": "핵심 슬로건",
  "tagline": "짧은 태그라인",
  "core_values": ["핵심가치1", "핵심가치2", ...],
  "brand_philosophy": "브랜드 철학 (2-3문장)",
  "target_keywords": ["타겟키워드1", "타겟키워드2", ...],
  "aesthetic_style": "Natural/Trendy/Luxury/Classic/Minimal 중 하나",
  "product_categories": ["제품카테고리1", "제품카테고리2", ...],
  "price_tier": "Premium/Mid-range/Professional/Mass 중 하나",
  "age_target": "타겟 연령층",
  "category": "Hair Care"
}}

JSON만 출력하세요."""


class BrandCrawler:
    """아모레퍼시픽 헤어 브랜드 크롤러"""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent / "data"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0'
        })

    def crawl_hair_brands(self) -> Dict:
        """헤어 브랜드 목록 크롤링"""
        try:
            response = self.session.get(HAIR_BRANDS_URL, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            brands = {}
            hair_brands = []

            # 브랜드 카드 파싱
            brand_cards = soup.select('.brand-card, .brand-item, [data-category="Hair"]')

            for card in brand_cards:
                brand_info = self._parse_brand_card(card)
                if brand_info:
                    name = brand_info['brand_name']
                    brands[name] = brand_info
                    hair_brands.append(name)

            return {
                "brands": brands,
                "hair_brands": hair_brands,
                "metadata": {
                    "source": HAIR_BRANDS_URL,
                    "last_updated": datetime.now().isoformat(),
                    "total_brands": len(brands)
                }
            }

        except Exception as e:
            logger.error(f"브랜드 크롤링 실패: {e}")
            return {"brands": {}, "hair_brands": [], "error": str(e)}

    def _parse_brand_card(self, card) -> Optional[Dict]:
        """브랜드 카드에서 정보 추출"""
        try:
            name_elem = card.select_one('.brand-name, h3, .title')
            name = name_elem.get_text(strip=True) if name_elem else None

            if not name:
                return None

            slogan_elem = card.select_one('.brand-slogan, .slogan, .description')
            slogan = slogan_elem.get_text(strip=True) if slogan_elem else ""

            link_elem = card.select_one('a[href]')
            detail_url = ""
            if link_elem:
                href = link_elem.get('href', '')
                detail_url = urljoin(BASE_URL, href)

            return {
                "brand_name": name,
                "slogan": slogan,
                "detail_url": detail_url
            }

        except Exception as e:
            logger.warning(f"브랜드 카드 파싱 실패: {e}")
            return None

    def enrich_with_llm(self, brand_data: Dict) -> Dict:
        """LLM으로 브랜드 정보 구조화"""
        if not OPENAI_AVAILABLE or not self.api_key:
            return brand_data

        try:
            client = openai.OpenAI(api_key=self.api_key)

            prompt = BRAND_EXTRACTION_PROMPT.format(
                brand_name=brand_data.get("brand_name", ""),
                slogan=brand_data.get("slogan", ""),
                description=brand_data.get("description", ""),
                detail_url=brand_data.get("detail_url", "")
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "브랜드 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.3
            )

            result_text = response.choices[0].message.content.strip()

            # JSON 파싱
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            enriched = json.loads(result_text.strip())
            enriched["detail_url"] = brand_data.get("detail_url", "")
            enriched["llm_structured"] = True

            return enriched

        except Exception as e:
            logger.warning(f"LLM 구조화 실패: {e}")
            return brand_data

    def crawl_and_save(self, use_llm: bool = True) -> Dict:
        """크롤링 후 파일 저장"""
        data = self.crawl_hair_brands()

        if use_llm and data.get("brands"):
            enriched_brands = {}
            for name, info in data["brands"].items():
                enriched = self.enrich_with_llm(info)
                enriched_brands[name] = enriched
                logger.info(f"브랜드 구조화 완료: {name}")

            data["brands"] = enriched_brands
            data["metadata"]["llm_structured"] = True

        # 저장
        output_path = self.data_dir / "amore_brands.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"브랜드 데이터 저장: {output_path}")
        return data

    def load_brands(self) -> Dict:
        """저장된 브랜드 데이터 로드"""
        path = self.data_dir / "amore_brands.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"brands": {}, "hair_brands": []}


# ============================================================
# Instagram 인플루언서 크롤러
# ============================================================

# 인플루언서 분석 프롬프트
INFLUENCER_ANALYSIS_PROMPT = """인플루언서의 이미지들을 분석하여 구조화된 JSON으로 추출하세요.

JSON 형식:
{{
  "dominant_style": "luxury/natural/trendy/colorful/minimal/professional 중 하나",
  "sub_styles": ["서브스타일1", "서브스타일2"],
  "color_palette": "warm/cool/neutral/vivid/muted",
  "aesthetic_tags": ["태그1", "태그2", "태그3"],
  "hair_style_tags": ["헤어스타일1", "헤어스타일2"],
  "vibe": "계정의 전반적인 분위기를 한 문장으로",
  "professionalism_score": 0.0-1.0,
  "trend_relevance_score": 0.0-1.0
}}

JSON만 출력하세요."""


class InfluencerCrawler:
    """
    Instagram Graph API 기반 인플루언서 크롤러

    Instagram Graph API를 통해 수집 가능한 데이터:
    - 기본 정보: username, followers_count, biography
    - 미디어: media_type, caption, like_count, comments_count, timestamp
    - 릴스: plays_count (조회수), thumbnail_url
    - 인사이트: audience_country (비즈니스 계정만)
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent / "data"
        self.access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
        self.business_account_id = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://graph.facebook.com/v18.0"

    def search_influencers(self, hashtag: str, limit: int = 30) -> List[Dict]:
        """
        해시태그 기반 인플루언서 검색

        Args:
            hashtag: 검색할 해시태그 (예: '헤어스타일', '미용사')
            limit: 최대 검색 수

        Returns:
            인플루언서 기본 정보 리스트
        """
        if not self.access_token or not self.business_account_id:
            logger.warning("Instagram API 토큰이 설정되지 않음. 샘플 데이터 사용.")
            return self._load_sample_data()

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
                logger.warning(f"해시태그 '{hashtag}' 검색 결과 없음")
                return []

            hashtag_id = hashtag_data["data"][0]["id"]

            # 2. 해시태그 관련 미디어 검색
            media_url = f"{self.base_url}/{hashtag_id}/top_media"
            params = {
                "user_id": self.business_account_id,
                "fields": "id,caption,media_type,permalink,timestamp,like_count,comments_count",
                "limit": limit,
                "access_token": self.access_token
            }
            response = requests.get(media_url, params=params, timeout=10)
            response.raise_for_status()
            media_data = response.json()

            # 3. 각 미디어에서 인플루언서 정보 추출
            influencers = []
            seen_users = set()

            for media in media_data.get("data", []):
                user_info = self._extract_user_from_media(media)
                if user_info and user_info["username"] not in seen_users:
                    seen_users.add(user_info["username"])
                    influencers.append(user_info)

            return influencers

        except Exception as e:
            logger.error(f"인플루언서 검색 실패: {e}")
            return self._load_sample_data()

    def get_influencer_details(self, username: str) -> Optional[Dict]:
        """
        특정 인플루언서의 상세 정보 수집

        Args:
            username: 인스타그램 사용자명

        Returns:
            인플루언서 상세 정보
        """
        if not self.access_token:
            logger.warning("Instagram API 토큰이 설정되지 않음")
            return None

        try:
            # 비즈니스 디스커버리 API로 타 계정 정보 조회
            url = f"{self.base_url}/{self.business_account_id}"
            params = {
                "fields": f"business_discovery.username({username})"
                          "{username,name,biography,followers_count,media_count,"
                          "media.limit(12){id,caption,like_count,comments_count,"
                          "media_type,timestamp,thumbnail_url,media_url}}",
                "access_token": self.access_token
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            discovery = data.get("business_discovery", {})
            if not discovery:
                return None

            # 데이터 정규화
            return self._normalize_influencer_data(discovery)

        except Exception as e:
            logger.error(f"인플루언서 상세 정보 수집 실패 ({username}): {e}")
            return None

    def _extract_user_from_media(self, media: Dict) -> Optional[Dict]:
        """미디어에서 사용자 정보 추출"""
        try:
            permalink = media.get("permalink", "")
            # permalink에서 username 추출: https://www.instagram.com/p/xxx/ 형식
            # 또는 https://www.instagram.com/username/p/xxx/ 형식

            return {
                "media_id": media.get("id"),
                "caption": media.get("caption", ""),
                "likes": media.get("like_count", 0),
                "comments": media.get("comments_count", 0),
                "timestamp": media.get("timestamp"),
                "username": None  # 추후 business_discovery로 보완 필요
            }
        except Exception:
            return None

    def _normalize_influencer_data(self, raw_data: Dict) -> Dict:
        """
        Instagram API 응답을 시스템 표준 스키마로 정규화

        표준 스키마:
        - username: 계정 ID
        - followers: 팔로워 수
        - bio: 자기소개
        - recent_posts: 최근 게시물 리스트
        - audience_countries: 국가별 오디언스 비율
        - avg_upload_interval_days: 평균 업로드 간격
        """
        username = raw_data.get("username", "")

        # 최근 게시물 정규화
        recent_posts = []
        media_list = raw_data.get("media", {}).get("data", [])

        timestamps = []
        for media in media_list:
            post = {
                "caption": media.get("caption", ""),
                "likes": media.get("like_count", 0),
                "comments": media.get("comments_count", 0),
                "views": media.get("plays_count", 0),  # 릴스인 경우
                "media_type": media.get("media_type", ""),
                "timestamp": media.get("timestamp", ""),
                "thumbnail_url": media.get("thumbnail_url", ""),
                "image_url": media.get("media_url", "")
            }
            recent_posts.append(post)

            if media.get("timestamp"):
                timestamps.append(media["timestamp"])

        # 업로드 간격 계산
        avg_interval = self._calculate_upload_interval(timestamps)

        return {
            "username": username,
            "followers": raw_data.get("followers_count", 0),
            "bio": raw_data.get("biography", ""),
            "media_count": raw_data.get("media_count", 0),
            "recent_posts": recent_posts,
            "audience_countries": {},  # 인사이트 API 별도 호출 필요
            "avg_upload_interval_days": avg_interval,
            "crawled_at": datetime.now().isoformat()
        }

    def _calculate_upload_interval(self, timestamps: List[str]) -> float:
        """타임스탬프 리스트로 평균 업로드 간격 계산"""
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

    def analyze_images_with_llm(self, influencer: Dict) -> Dict:
        """
        LLM 비전으로 인플루언서 이미지 분석

        Args:
            influencer: 인플루언서 데이터 (recent_posts 포함)

        Returns:
            이미지 분석 결과 추가된 인플루언서 데이터
        """
        if not OPENAI_AVAILABLE or not self.api_key:
            return influencer

        # 썸네일 URL 수집 (최대 5개)
        image_urls = []
        for post in influencer.get("recent_posts", [])[:5]:
            url = post.get("thumbnail_url") or post.get("image_url")
            if url:
                image_urls.append(url)

        if not image_urls:
            return influencer

        try:
            client = openai.OpenAI(api_key=self.api_key)

            # 이미지 분석 요청
            content = [{"type": "text", "text": "다음 인플루언서의 이미지들을 분석하세요."}]
            for url in image_urls[:3]:  # 최대 3개
                content.append({"type": "image_url", "image_url": {"url": url}})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": INFLUENCER_ANALYSIS_PROMPT},
                    {"role": "user", "content": content}
                ],
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()

            # JSON 파싱
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            image_analysis = json.loads(result_text.strip())
            influencer["image_analysis"] = image_analysis

        except Exception as e:
            logger.warning(f"이미지 분석 실패 ({influencer.get('username', '')}): {e}")

        return influencer

    def _load_sample_data(self) -> List[Dict]:
        """샘플 데이터 로드 (API 미설정 시)"""
        path = self.data_dir / "influencers_data.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("influencers", [])
        return []

    def crawl_and_save(self, hashtags: List[str] = None, use_llm: bool = True) -> Dict:
        """
        인플루언서 크롤링 및 저장

        Args:
            hashtags: 검색할 해시태그 리스트
            use_llm: LLM 이미지 분석 사용 여부

        Returns:
            크롤링 결과
        """
        if hashtags is None:
            hashtags = ["헤어스타일", "미용사", "헤어디자이너", "살롱"]

        all_influencers = []
        seen_usernames = set()

        for hashtag in hashtags:
            logger.info(f"해시태그 검색 중: #{hashtag}")
            influencers = self.search_influencers(hashtag)

            for inf in influencers:
                username = inf.get("username")
                if username and username not in seen_usernames:
                    seen_usernames.add(username)

                    # 상세 정보 수집
                    details = self.get_influencer_details(username)
                    if details:
                        # LLM 이미지 분석
                        if use_llm:
                            details = self.analyze_images_with_llm(details)
                        all_influencers.append(details)

        # 저장
        result = {
            "influencers": all_influencers,
            "metadata": {
                "hashtags": hashtags,
                "total_count": len(all_influencers),
                "crawled_at": datetime.now().isoformat()
            }
        }

        output_path = self.data_dir / "influencers_data.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"인플루언서 데이터 저장: {output_path} ({len(all_influencers)}명)")
        return result

    def load_influencers(self) -> List[Dict]:
        """저장된 인플루언서 데이터 로드"""
        path = self.data_dir / "influencers_data.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("influencers", [])
        return []


# 테스트
if __name__ == "__main__":
    # 브랜드 크롤러 테스트
    brand_crawler = BrandCrawler()
    brands = brand_crawler.load_brands()
    print(f"로드된 브랜드: {len(brands.get('brands', {}))}")

    # 인플루언서 크롤러 테스트
    inf_crawler = InfluencerCrawler()
    influencers = inf_crawler.load_influencers()
    print(f"로드된 인플루언서: {len(influencers)}")
