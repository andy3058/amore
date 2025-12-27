"""
인플루언서 데이터 수집 서비스
Instagram Graph API를 활용한 인플루언서 프로필 및 게시물 수집
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from modules.instagram_api import (
    InstagramGraphAPI,
    InstagramAPIError,
    RateLimitError,
    get_instagram_api
)

try:
    from config.instagram import HAIR_HASHTAGS
except ImportError:
    HAIR_HASHTAGS = ["헤어스타일", "헤어케어", "미용사", "헤어디자이너"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InfluencerScraperService:
    """
    인플루언서 데이터 수집 서비스

    사용법:
        scraper = InfluencerScraperService()

        # 특정 인플루언서 수집
        data = scraper.scrape_influencer("username")

        # 해시태그로 인플루언서 발굴
        influencers = scraper.discover_by_hashtags(["헤어스타일", "헤어케어"])

        # 기존 데이터 업데이트
        scraper.update_all_influencers()
    """

    def __init__(self, api: InstagramGraphAPI = None):
        """
        서비스 초기화

        Args:
            api: InstagramGraphAPI 인스턴스 (None이면 자동 생성)
        """
        self.api = api
        self._api_available = False

        # API 초기화 시도
        try:
            if self.api is None:
                self.api = get_instagram_api()
            self._api_available = True
        except InstagramAPIError as e:
            logger.warning(f"Instagram API 초기화 실패: {e}")
            logger.info("API 없이 오프라인 모드로 동작합니다.")

        # 데이터 파일 경로
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_file = self.data_dir / "influencers_data.json"

    def is_api_available(self) -> bool:
        """API 사용 가능 여부 확인"""
        return self._api_available

    def load_existing_data(self) -> Dict:
        """기존 인플루언서 데이터 로드"""
        if self.data_file.exists():
            with open(self.data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"influencers": [], "last_updated": None}

    def save_data(self, data: Dict):
        """인플루언서 데이터 저장"""
        data["last_updated"] = datetime.now().isoformat()

        # 디렉토리 생성
        self.data_dir.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"데이터 저장 완료: {len(data['influencers'])}명")

    def scrape_influencer(self, username: str) -> Optional[Dict]:
        """
        특정 인플루언서 프로필 수집

        Args:
            username: Instagram 사용자명 (@제외)

        Returns:
            인플루언서 데이터 또는 None
        """
        if not self._api_available:
            logger.error("API가 초기화되지 않았습니다.")
            return None

        try:
            logger.info(f"인플루언서 수집 중: @{username}")
            profile = self.api.get_influencer_profile(username)

            if profile:
                logger.info(
                    f"수집 완료: @{username} "
                    f"(팔로워: {profile.get('followers', 0):,}명, "
                    f"게시물: {len(profile.get('recent_posts', []))}개)"
                )
            else:
                logger.warning(f"프로필을 찾을 수 없음: @{username}")

            return profile

        except RateLimitError as e:
            logger.error(f"Rate limit 초과: {e}. {e.retry_after}초 후 재시도 필요")
            raise
        except InstagramAPIError as e:
            logger.error(f"API 오류: {e}")
            return None

    def scrape_multiple(
        self,
        usernames: List[str],
        skip_existing: bool = True
    ) -> Tuple[List[Dict], List[str]]:
        """
        여러 인플루언서 프로필 수집

        Args:
            usernames: 사용자명 리스트
            skip_existing: True면 이미 수집된 사용자 스킵

        Returns:
            (수집된 프로필 리스트, 실패한 사용자명 리스트)
        """
        if not self._api_available:
            logger.error("API가 초기화되지 않았습니다.")
            return [], usernames

        # 기존 데이터 로드
        existing_data = self.load_existing_data()
        existing_usernames = {
            inf.get("username") for inf in existing_data.get("influencers", [])
        }

        collected = []
        failed = []

        for username in usernames:
            # 이미 존재하는 경우 스킵
            if skip_existing and username in existing_usernames:
                logger.info(f"스킵 (이미 존재): @{username}")
                continue

            try:
                profile = self.scrape_influencer(username)
                if profile:
                    collected.append(profile)
                else:
                    failed.append(username)

            except RateLimitError:
                # Rate limit 도달 시 중단
                logger.warning("Rate limit으로 인해 수집 중단")
                failed.extend(usernames[usernames.index(username):])
                break

        return collected, failed

    def discover_by_hashtags(
        self,
        hashtags: List[str] = None,
        min_likes: int = 500,
        min_followers: int = 10000,
        limit_per_hashtag: int = 30
    ) -> List[Dict]:
        """
        해시태그로 인플루언서 발굴

        Args:
            hashtags: 검색할 해시태그 리스트 (None이면 기본 헤어 해시태그 사용)
            min_likes: 게시물 최소 좋아요 수
            min_followers: 최소 팔로워 수
            limit_per_hashtag: 해시태그당 검색할 게시물 수

        Returns:
            발굴된 인플루언서 리스트
        """
        if not self._api_available:
            logger.error("API가 초기화되지 않았습니다.")
            return []

        hashtags = hashtags or HAIR_HASHTAGS[:5]  # 주간 제한으로 5개만

        discovered_usernames = set()
        discovered_influencers = []

        for hashtag in hashtags:
            try:
                logger.info(f"해시태그 검색 중: #{hashtag}")
                posts = self.api.search_hashtag(hashtag, "top", limit_per_hashtag)

                for post in posts:
                    if post.get("like_count", 0) < min_likes:
                        continue

                    # 게시물에서 사용자 정보 추출 시도
                    # 주의: 해시태그 검색 결과에서는 사용자명을 직접 얻기 어려움
                    # permalink에서 추출 시도
                    permalink = post.get("permalink", "")
                    # permalink 예: https://www.instagram.com/p/ABC123/
                    # 사용자명은 이 API로 직접 얻을 수 없음

                logger.info(f"#{hashtag}: {len(posts)}개 게시물 검색됨")

            except InstagramAPIError as e:
                logger.error(f"해시태그 #{hashtag} 검색 실패: {e}")
                continue

        return discovered_influencers

    def add_influencer(self, username: str) -> Optional[Dict]:
        """
        인플루언서를 수집하고 데이터에 추가

        Args:
            username: Instagram 사용자명

        Returns:
            추가된 인플루언서 데이터
        """
        profile = self.scrape_influencer(username)

        if profile:
            data = self.load_existing_data()

            # 중복 체크
            existing_usernames = {
                inf.get("username") for inf in data.get("influencers", [])
            }

            if username not in existing_usernames:
                data["influencers"].append(profile)
                self.save_data(data)
                logger.info(f"인플루언서 추가됨: @{username}")
            else:
                # 기존 데이터 업데이트
                for i, inf in enumerate(data["influencers"]):
                    if inf.get("username") == username:
                        data["influencers"][i] = profile
                        break
                self.save_data(data)
                logger.info(f"인플루언서 업데이트됨: @{username}")

        return profile

    def update_influencer(self, username: str) -> Optional[Dict]:
        """
        기존 인플루언서 데이터 업데이트

        Args:
            username: Instagram 사용자명

        Returns:
            업데이트된 인플루언서 데이터
        """
        return self.add_influencer(username)  # 동일 로직

    def update_all_influencers(self, max_updates: int = 50) -> Tuple[int, int]:
        """
        모든 인플루언서 데이터 업데이트

        Args:
            max_updates: 최대 업데이트 수 (Rate limit 고려)

        Returns:
            (성공 수, 실패 수)
        """
        if not self._api_available:
            logger.error("API가 초기화되지 않았습니다.")
            return 0, 0

        data = self.load_existing_data()
        influencers = data.get("influencers", [])

        success_count = 0
        fail_count = 0

        for i, inf in enumerate(influencers[:max_updates]):
            username = inf.get("username")
            if not username:
                continue

            try:
                updated = self.scrape_influencer(username)
                if updated:
                    data["influencers"][i] = updated
                    success_count += 1
                else:
                    fail_count += 1

            except RateLimitError:
                logger.warning("Rate limit으로 인해 업데이트 중단")
                break

        if success_count > 0:
            self.save_data(data)

        logger.info(f"업데이트 완료: 성공 {success_count}, 실패 {fail_count}")
        return success_count, fail_count

    def remove_influencer(self, username: str) -> bool:
        """
        인플루언서 삭제

        Args:
            username: 삭제할 사용자명

        Returns:
            삭제 성공 여부
        """
        data = self.load_existing_data()
        original_count = len(data.get("influencers", []))

        data["influencers"] = [
            inf for inf in data.get("influencers", [])
            if inf.get("username") != username
        ]

        if len(data["influencers"]) < original_count:
            self.save_data(data)
            logger.info(f"인플루언서 삭제됨: @{username}")
            return True

        return False

    def get_statistics(self) -> Dict:
        """데이터 통계 반환"""
        data = self.load_existing_data()
        influencers = data.get("influencers", [])

        if not influencers:
            return {
                "total_count": 0,
                "last_updated": data.get("last_updated"),
                "api_available": self._api_available
            }

        followers = [inf.get("followers", 0) for inf in influencers]
        posts_counts = [len(inf.get("recent_posts", [])) for inf in influencers]

        # API로 수집된 비율
        api_scraped = sum(
            1 for inf in influencers
            if inf.get("_source") == "instagram_api"
        )

        return {
            "total_count": len(influencers),
            "api_scraped_count": api_scraped,
            "manual_count": len(influencers) - api_scraped,
            "avg_followers": sum(followers) / len(followers) if followers else 0,
            "max_followers": max(followers) if followers else 0,
            "min_followers": min(followers) if followers else 0,
            "avg_posts": sum(posts_counts) / len(posts_counts) if posts_counts else 0,
            "last_updated": data.get("last_updated"),
            "api_available": self._api_available,
            "rate_limit_status": self.api.get_rate_limit_status() if self._api_available else None
        }

    def search_by_username(self, query: str) -> List[Dict]:
        """
        사용자명으로 기존 데이터 검색

        Args:
            query: 검색어

        Returns:
            매칭되는 인플루언서 리스트
        """
        data = self.load_existing_data()
        influencers = data.get("influencers", [])

        query_lower = query.lower()
        return [
            inf for inf in influencers
            if query_lower in inf.get("username", "").lower()
            or query_lower in inf.get("bio", "").lower()
        ]


# 싱글톤 인스턴스
_scraper_instance: Optional[InfluencerScraperService] = None


def get_scraper_service() -> InfluencerScraperService:
    """스크래퍼 서비스 싱글톤 인스턴스 반환"""
    global _scraper_instance

    if _scraper_instance is None:
        _scraper_instance = InfluencerScraperService()

    return _scraper_instance


# CLI 테스트
if __name__ == "__main__":
    import sys

    scraper = InfluencerScraperService()

    print("\n=== 인플루언서 스크래퍼 ===")
    print(f"API 사용 가능: {scraper.is_api_available()}")

    # 통계 출력
    stats = scraper.get_statistics()
    print(f"\n현재 데이터:")
    print(f"  - 총 인플루언서: {stats['total_count']}명")
    print(f"  - API 수집: {stats.get('api_scraped_count', 0)}명")
    print(f"  - 평균 팔로워: {stats.get('avg_followers', 0):,.0f}명")
    print(f"  - 마지막 업데이트: {stats.get('last_updated', 'N/A')}")

    if scraper.is_api_available() and len(sys.argv) > 1:
        username = sys.argv[1]
        print(f"\n@{username} 프로필 수집 중...")
        result = scraper.scrape_influencer(username)
        if result:
            print(json.dumps(result, ensure_ascii=False, indent=2))