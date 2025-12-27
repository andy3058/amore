"""
아모레퍼시픽 브랜드 크롤러
공식 사이트에서 브랜드 정보를 수집하여 데이터 업데이트
"""
import json
import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== 설정 ==============

BASE_URL = "https://www.apgroup.com"
BRANDS_LIST_URL = f"{BASE_URL}/int/ko/brands/brands.html"

# 카테고리 매핑 (페이지에서 추출이 어려운 경우 수동 매핑)
CATEGORY_MAPPING = {
    "설화수": "Beauty Care",
    "라네즈": "Beauty Care",
    "이니스프리": "Beauty Care",
    "헤라": "Beauty Care",
    "프리메라": "Beauty Care",
    "아이오페": "Medical Beauty",
    "마몽드": "Beauty Care",
    "한율": "Beauty Care",
    "에스트라": "Medical Beauty",
    "에스쁘아": "Makeup",
    "에뛰드": "Makeup",
    "려": "Hair Care",
    "미쟝센": "Hair Care",
    "라보에이치": "Hair Care",
    "아윤채": "Hair Care",
    "아모스프로페셔널": "Hair Care",
    "롱테이크": "Fragrance",
    "일리윤": "Beauty Care",
    "해피바스": "Body Care",
    "스킨유": "Body Care",
    "메디안": "Oral Care",
    "젠티스트": "Oral Care",
    "바이탈뷰티": "Inner Beauty",
    "오설록": "Tea Culture",
    "메이크온": "Beauty Device",
    "오딧세이": "Men's Beauty",
    "비레디": "Beauty Care",
    "홀리추얼": "Beauty Care",
    "타타하퍼": "Beauty Care",
    "코스알엑스": "Beauty Care",
    "에이피뷰티": "Beauty Care",
}

# 스타일 매핑 (브랜드별 기본 스타일)
STYLE_MAPPING = {
    "설화수": "Luxury",
    "라네즈": "Trendy",
    "이니스프리": "Natural",
    "헤라": "Luxury",
    "프리메라": "Natural",
    "아이오페": "Minimal",
    "마몽드": "Natural",
    "한율": "Classic",
    "에스트라": "Minimal",
    "에스쁘아": "Trendy",
    "에뛰드": "Colorful",
    "려": "Classic",
    "미쟝센": "Trendy",
    "라보에이치": "Minimal",
    "아윤채": "Luxury",
    "아모스프로페셔널": "Classic",
    "일리윤": "Natural",
    "해피바스": "Mass",
    "바이탈뷰티": "Minimal",
    "오설록": "Natural",
    "롱테이크": "Minimal",
    "메디안": "Mass",
    "코스알엑스": "Minimal",
    "타타하퍼": "Luxury",
}


@dataclass
class BrandInfo:
    """브랜드 정보 데이터 클래스"""
    brand_name: str
    brand_name_en: str
    slogan: str = ""
    tagline: str = ""
    core_values: List[str] = None
    brand_philosophy: str = ""
    target_keywords: List[str] = None
    aesthetic_style: str = ""
    product_categories: List[str] = None
    price_tier: str = ""
    age_target: str = ""
    category: str = ""
    detail_url: str = ""
    image_url: str = ""

    def __post_init__(self):
        if self.core_values is None:
            self.core_values = []
        if self.target_keywords is None:
            self.target_keywords = []
        if self.product_categories is None:
            self.product_categories = []

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v}


class AmorePacificBrandCrawler:
    """
    아모레퍼시픽 브랜드 크롤러

    사용법:
        crawler = AmorePacificBrandCrawler()

        # 브랜드 목록 크롤링
        brands = crawler.crawl_brand_list()

        # 특정 브랜드 상세 정보 크롤링
        detail = crawler.crawl_brand_detail("려", "/int/ko/brands/ryo.html")

        # 전체 크롤링 및 저장
        crawler.crawl_all_and_save()
    """

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        })

        # 데이터 파일 경로
        self.data_dir = Path(__file__).parent.parent / "data"
        self.output_file = self.data_dir / "amore_brands.json"

    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """페이지 HTML 가져오기"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            logger.error(f"페이지 로드 실패: {url} - {e}")
            return None

    def _extract_korean_name(self, text: str) -> Tuple[str, str]:
        """
        텍스트에서 한글 이름과 영문 이름 분리
        예: "설화수 (Sulwhasoo)" -> ("설화수", "Sulwhasoo")
        """
        # 괄호 안의 영문 이름 추출
        match = re.search(r"(.+?)\s*[\(（]([A-Za-z\s\-]+)[\)）]", text)
        if match:
            return match.group(1).strip(), match.group(2).strip()

        # 괄호가 없으면 전체를 한글 이름으로
        return text.strip(), ""

    def crawl_brand_list(self) -> List[Dict]:
        """
        브랜드 목록 페이지에서 모든 브랜드 기본 정보 크롤링

        Returns:
            브랜드 기본 정보 리스트
        """
        logger.info(f"브랜드 목록 크롤링 시작: {BRANDS_LIST_URL}")

        soup = self._fetch_page(BRANDS_LIST_URL)
        if not soup:
            return []

        brands = []

        # 브랜드 카드 요소 찾기 (여러 선택자 시도)
        brand_cards = soup.select(".brand-card, .brand-item, .brands-list li, [class*='brand']")

        if not brand_cards:
            # 대체 방법: 링크에서 브랜드 추출
            brand_links = soup.select("a[href*='/brands/'][href$='.html']")
            for link in brand_links:
                href = link.get("href", "")
                if "/brands/brands.html" in href:
                    continue

                name_text = link.get_text(strip=True)
                if not name_text:
                    # 이미지 alt 텍스트에서 추출
                    img = link.select_one("img")
                    if img:
                        name_text = img.get("alt", "")

                if name_text:
                    korean_name, english_name = self._extract_korean_name(name_text)

                    # tagline 추출 시도
                    tagline = ""
                    parent = link.parent
                    if parent:
                        tagline_elem = parent.select_one(".tagline, .description, .sub-title, p")
                        if tagline_elem:
                            tagline = tagline_elem.get_text(strip=True)

                    brands.append({
                        "name": korean_name,
                        "name_en": english_name,
                        "tagline": tagline,
                        "detail_url": urljoin(self.base_url, href),
                        "category": CATEGORY_MAPPING.get(korean_name, "Beauty Care")
                    })

        else:
            for card in brand_cards:
                # 브랜드 이름
                name_elem = card.select_one(".brand-name, .name, h3, h4, strong")
                if not name_elem:
                    continue

                name_text = name_elem.get_text(strip=True)
                korean_name, english_name = self._extract_korean_name(name_text)

                # 상세 페이지 링크
                link = card.select_one("a[href*='/brands/']")
                detail_url = ""
                if link:
                    detail_url = urljoin(self.base_url, link.get("href", ""))

                # tagline
                tagline_elem = card.select_one(".tagline, .description, .sub-title")
                tagline = tagline_elem.get_text(strip=True) if tagline_elem else ""

                # 이미지 URL
                img = card.select_one("img")
                image_url = ""
                if img:
                    image_url = urljoin(self.base_url, img.get("src", ""))

                brands.append({
                    "name": korean_name,
                    "name_en": english_name,
                    "tagline": tagline,
                    "detail_url": detail_url,
                    "image_url": image_url,
                    "category": CATEGORY_MAPPING.get(korean_name, "Beauty Care")
                })

        logger.info(f"브랜드 목록 크롤링 완료: {len(brands)}개")
        return brands

    def crawl_brand_detail(self, brand_name: str, detail_url: str) -> Optional[BrandInfo]:
        """
        브랜드 상세 페이지에서 정보 크롤링

        Args:
            brand_name: 브랜드 한글 이름
            detail_url: 상세 페이지 URL

        Returns:
            BrandInfo 객체 또는 None
        """
        logger.info(f"브랜드 상세 크롤링: {brand_name} - {detail_url}")

        soup = self._fetch_page(detail_url)
        if not soup:
            return None

        # 슬로건/tagline 추출
        slogan = ""
        tagline_selectors = [
            ".brand-slogan", ".slogan", ".tagline",
            ".hero-text", ".main-copy", ".brand-copy",
            "h2", ".subtitle"
        ]
        for selector in tagline_selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                if text and len(text) < 200:  # 너무 긴 텍스트 제외
                    slogan = text
                    break

        # 브랜드 철학/설명 추출
        philosophy = ""
        desc_selectors = [
            ".brand-description", ".brand-story", ".brand-philosophy",
            ".about-brand", ".intro-text", "article p", ".content p"
        ]
        for selector in desc_selectors:
            elems = soup.select(selector)
            for elem in elems:
                text = elem.get_text(strip=True)
                if text and 20 < len(text) < 500:
                    philosophy = text
                    break
            if philosophy:
                break

        # 영문 이름 추출
        english_name = ""
        # URL에서 추출
        url_match = re.search(r"/brands/([^/]+)\.html", detail_url)
        if url_match:
            english_name = url_match.group(1).replace("-", " ").title()

        # 페이지에서 추출 시도
        en_name_elem = soup.select_one(".brand-name-en, .english-name, [lang='en']")
        if en_name_elem:
            english_name = en_name_elem.get_text(strip=True)

        # 키워드 추출 (메타 태그 또는 텍스트에서)
        keywords = []
        meta_keywords = soup.select_one("meta[name='keywords']")
        if meta_keywords:
            keywords = [k.strip() for k in meta_keywords.get("content", "").split(",")]

        # 카테고리
        category = CATEGORY_MAPPING.get(brand_name, "Beauty Care")

        # 스타일
        style = STYLE_MAPPING.get(brand_name, "Trendy")

        return BrandInfo(
            brand_name=brand_name,
            brand_name_en=english_name,
            slogan=slogan,
            tagline=slogan,
            brand_philosophy=philosophy,
            target_keywords=keywords[:10],  # 최대 10개
            aesthetic_style=style,
            category=category,
            detail_url=detail_url
        )

    def crawl_all_brands(self, include_details: bool = True) -> Dict[str, BrandInfo]:
        """
        모든 브랜드 정보 크롤링

        Args:
            include_details: True면 각 브랜드 상세 페이지도 크롤링

        Returns:
            브랜드명을 키로 하는 BrandInfo 딕셔너리
        """
        # 브랜드 목록 가져오기
        brand_list = self.crawl_brand_list()

        brands = {}

        for brand_data in brand_list:
            name = brand_data["name"]

            if include_details and brand_data.get("detail_url"):
                # 상세 페이지 크롤링
                detail = self.crawl_brand_detail(name, brand_data["detail_url"])
                if detail:
                    # 목록에서 가져온 정보로 보완
                    if not detail.brand_name_en and brand_data.get("name_en"):
                        detail.brand_name_en = brand_data["name_en"]
                    if not detail.tagline and brand_data.get("tagline"):
                        detail.tagline = brand_data["tagline"]
                        detail.slogan = brand_data["tagline"]

                    brands[name] = detail
            else:
                # 목록 정보만으로 생성
                brands[name] = BrandInfo(
                    brand_name=name,
                    brand_name_en=brand_data.get("name_en", ""),
                    tagline=brand_data.get("tagline", ""),
                    slogan=brand_data.get("tagline", ""),
                    category=brand_data.get("category", "Beauty Care"),
                    aesthetic_style=STYLE_MAPPING.get(name, "Trendy"),
                    detail_url=brand_data.get("detail_url", "")
                )

        logger.info(f"전체 브랜드 크롤링 완료: {len(brands)}개")
        return brands

    def load_existing_data(self) -> Dict:
        """기존 브랜드 데이터 로드"""
        if self.output_file.exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"brands": {}, "hair_brands": [], "metadata": {}}

    def merge_with_existing(
        self,
        crawled: Dict[str, BrandInfo],
        existing: Dict
    ) -> Dict:
        """
        크롤링 데이터와 기존 데이터 병합
        기존 데이터의 수동 입력 정보는 보존하면서 크롤링 데이터로 업데이트

        Args:
            crawled: 크롤링된 브랜드 정보
            existing: 기존 브랜드 데이터

        Returns:
            병합된 데이터
        """
        existing_brands = existing.get("brands", {})

        merged_brands = {}

        for name, brand_info in crawled.items():
            if name in existing_brands:
                # 기존 데이터가 있으면 병합
                old = existing_brands[name]
                new = brand_info.to_dict()

                # 기존 수동 입력 데이터 보존 (크롤링으로 얻기 어려운 필드)
                preserve_fields = [
                    "core_values", "product_categories", "product_lines",
                    "price_tier", "age_target", "market_position", "launch_year"
                ]

                for field in preserve_fields:
                    if field in old and old[field]:
                        new[field] = old[field]

                # 크롤링 데이터가 비어있으면 기존 데이터 사용
                for key, value in old.items():
                    if key not in new or not new.get(key):
                        new[key] = value

                merged_brands[name] = new
            else:
                # 새 브랜드
                merged_brands[name] = brand_info.to_dict()

        # 기존에만 있는 브랜드 보존
        for name, data in existing_brands.items():
            if name not in merged_brands:
                merged_brands[name] = data

        # 카테고리별 분류
        hair_brands = [
            name for name, data in merged_brands.items()
            if data.get("category") == "Hair Care"
        ]

        skincare_brands = [
            name for name, data in merged_brands.items()
            if data.get("category") in ["Beauty Care", "Medical Beauty"]
        ]

        makeup_brands = [
            name for name, data in merged_brands.items()
            if data.get("category") == "Makeup"
        ]

        # 전체 카테고리 정리
        categories = {}
        for name, data in merged_brands.items():
            cat = data.get("category", "Other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)

        return {
            "brands": merged_brands,
            "hair_brands": hair_brands,
            "skincare_brands": skincare_brands,
            "makeup_brands": makeup_brands,
            "categories": categories,
            "metadata": {
                "source": f"아모레퍼시픽 공식 사이트 ({BRANDS_LIST_URL})",
                "last_updated": datetime.now().isoformat(),
                "total_brands": len(merged_brands),
                "version": existing.get("metadata", {}).get("version", "1.0"),
                "crawled_date": datetime.now().strftime("%Y-%m-%d")
            }
        }

    def save_data(self, data: Dict):
        """데이터 저장"""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"데이터 저장 완료: {self.output_file}")

    def crawl_all_and_save(self, include_details: bool = False) -> Dict:
        """
        전체 크롤링 및 저장

        Args:
            include_details: True면 각 브랜드 상세 페이지도 크롤링 (시간이 오래 걸림)

        Returns:
            저장된 데이터
        """
        logger.info("=== 아모레퍼시픽 브랜드 크롤링 시작 ===")

        # 크롤링
        crawled = self.crawl_all_brands(include_details=include_details)

        # 기존 데이터 로드
        existing = self.load_existing_data()

        # 병합
        merged = self.merge_with_existing(crawled, existing)

        # 저장
        self.save_data(merged)

        logger.info("=== 크롤링 완료 ===")
        logger.info(f"총 브랜드 수: {len(merged['brands'])}")
        logger.info(f"헤어 브랜드: {merged['hair_brands']}")

        return merged


def crawl_brands(include_details: bool = False, save: bool = True) -> Dict:
    """
    브랜드 크롤링 헬퍼 함수

    Args:
        include_details: 상세 페이지 크롤링 여부
        save: 파일 저장 여부

    Returns:
        크롤링된 브랜드 데이터
    """
    crawler = AmorePacificBrandCrawler()

    if save:
        return crawler.crawl_all_and_save(include_details=include_details)
    else:
        crawled = crawler.crawl_all_brands(include_details=include_details)
        existing = crawler.load_existing_data()
        return crawler.merge_with_existing(crawled, existing)


# CLI 실행
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="아모레퍼시픽 브랜드 크롤러")
    parser.add_argument(
        "--details",
        action="store_true",
        help="각 브랜드 상세 페이지도 크롤링 (시간 오래 걸림)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="파일 저장하지 않고 결과만 출력"
    )

    args = parser.parse_args()

    result = crawl_brands(
        include_details=args.details,
        save=not args.no_save
    )

    print(f"\n총 {len(result['brands'])}개 브랜드 크롤링됨")
    print(f"헤어 브랜드: {result['hair_brands']}")

    if args.no_save:
        print("\n=== 크롤링 결과 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))