"""
Instagram Graph API 클라이언트
공식 Instagram Graph API를 통한 데이터 수집
"""
import os
import time
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 설정 임포트
try:
    from config.instagram import (
        INSTAGRAM_API_VERSION,
        INSTAGRAM_GRAPH_API_BASE_URL,
        FACEBOOK_GRAPH_API_BASE_URL,
        INSTAGRAM_ACCESS_TOKEN,
        INSTAGRAM_BUSINESS_ACCOUNT_ID,
        FACEBOOK_APP_SECRET,
        PROFILE_FIELDS,
        MEDIA_FIELDS,
        BUSINESS_DISCOVERY_FIELDS,
        BUSINESS_DISCOVERY_MEDIA_FIELDS,
        RATE_LIMIT_PER_HOUR,
        HASHTAG_SEARCH_LIMIT_PER_WEEK
    )
except ImportError:
    # 기본값 설정
    INSTAGRAM_API_VERSION = "v21.0"
    INSTAGRAM_GRAPH_API_BASE_URL = f"https://graph.instagram.com/{INSTAGRAM_API_VERSION}"
    FACEBOOK_GRAPH_API_BASE_URL = f"https://graph.facebook.com/{INSTAGRAM_API_VERSION}"
    INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
    INSTAGRAM_BUSINESS_ACCOUNT_ID = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID")
    FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET")
    PROFILE_FIELDS = ["username", "name", "biography", "followers_count", "media_count", "profile_picture_url"]
    MEDIA_FIELDS = ["id", "caption", "media_type", "timestamp", "like_count", "comments_count"]
    BUSINESS_DISCOVERY_FIELDS = ["username", "name", "biography", "followers_count", "media_count"]
    BUSINESS_DISCOVERY_MEDIA_FIELDS = ["id", "caption", "media_type", "timestamp", "like_count", "comments_count"]
    RATE_LIMIT_PER_HOUR = 200
    HASHTAG_SEARCH_LIMIT_PER_WEEK = 30

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstagramAPIError(Exception):
    """Instagram API 에러"""
    def __init__(self, message: str, error_code: int = None, error_subcode: int = None):
        super().__init__(message)
        self.error_code = error_code
        self.error_subcode = error_subcode


class RateLimitError(InstagramAPIError):
    """Rate Limit 초과 에러"""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message, error_code=429)
        self.retry_after = retry_after


@dataclass
class RateLimitStatus:
    """Rate Limit 상태"""
    call_count: int = 0
    total_time: int = 0
    total_cputime: int = 0
    percentage: float = 0.0
    estimated_reset: int = 0


class InstagramGraphAPI:
    """
    Instagram Graph API 클라이언트

    사용법:
        api = InstagramGraphAPI(access_token="your_token", business_account_id="your_id")

        # 다른 비즈니스 계정 프로필 조회
        profile = api.get_business_discovery("target_username")

        # 해시태그로 인플루언서 검색
        posts = api.search_hashtag("헤어스타일", limit=50)
    """

    def __init__(
        self,
        access_token: str = None,
        business_account_id: str = None,
        app_secret: str = None
    ):
        """
        Instagram Graph API 클라이언트 초기화

        Args:
            access_token: Instagram/Facebook 액세스 토큰
            business_account_id: 자신의 Instagram 비즈니스 계정 ID
            app_secret: Facebook 앱 시크릿 (토큰 갱신용)
        """
        self.access_token = access_token or INSTAGRAM_ACCESS_TOKEN
        self.business_account_id = business_account_id or INSTAGRAM_BUSINESS_ACCOUNT_ID
        self.app_secret = app_secret or FACEBOOK_APP_SECRET

        if not self.access_token:
            raise InstagramAPIError(
                "Access Token이 필요합니다. "
                "환경변수 INSTAGRAM_ACCESS_TOKEN을 설정하거나 생성자에 전달하세요."
            )

        if not self.business_account_id:
            raise InstagramAPIError(
                "Business Account ID가 필요합니다. "
                "환경변수 INSTAGRAM_BUSINESS_ACCOUNT_ID를 설정하거나 생성자에 전달하세요."
            )

        # HTTP 세션 설정 (재시도 로직 포함)
        self.session = self._create_session()

        # Rate Limit 추적
        self.rate_limit_status = RateLimitStatus()
        self._call_timestamps: List[float] = []

        # 해시태그 검색 추적 (주당 30개 제한)
        self._hashtag_searches: Dict[str, datetime] = {}

        # 캐시
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _create_session(self) -> requests.Session:
        """재시도 로직이 포함된 HTTP 세션 생성"""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _check_rate_limit(self):
        """Rate Limit 체크 및 필요시 대기"""
        now = time.time()
        hour_ago = now - 3600

        # 1시간 이내 호출만 유지
        self._call_timestamps = [ts for ts in self._call_timestamps if ts > hour_ago]

        if len(self._call_timestamps) >= RATE_LIMIT_PER_HOUR:
            oldest_call = min(self._call_timestamps)
            wait_time = oldest_call + 3600 - now
            if wait_time > 0:
                logger.warning(f"Rate limit 도달. {wait_time:.0f}초 대기...")
                raise RateLimitError(
                    f"시간당 {RATE_LIMIT_PER_HOUR}회 호출 제한에 도달했습니다.",
                    retry_after=int(wait_time)
                )

        self._call_timestamps.append(now)

    def _parse_rate_limit_headers(self, headers: Dict):
        """응답 헤더에서 Rate Limit 정보 파싱"""
        usage_header = headers.get("X-App-Usage") or headers.get("x-app-usage")
        if usage_header:
            try:
                usage = json.loads(usage_header)
                self.rate_limit_status.call_count = usage.get("call_count", 0)
                self.rate_limit_status.total_time = usage.get("total_time", 0)
                self.rate_limit_status.total_cputime = usage.get("total_cputime", 0)

                # 가장 높은 사용률을 percentage로 설정
                self.rate_limit_status.percentage = max(
                    usage.get("call_count", 0),
                    usage.get("total_time", 0),
                    usage.get("total_cputime", 0)
                )

                if self.rate_limit_status.percentage > 80:
                    logger.warning(f"Rate limit 사용량 높음: {self.rate_limit_status.percentage}%")
            except json.JSONDecodeError:
                pass

    def _request(
        self,
        method: str,
        url: str,
        params: Dict = None,
        data: Dict = None
    ) -> Dict:
        """
        API 요청 수행

        Args:
            method: HTTP 메서드
            url: 요청 URL
            params: 쿼리 파라미터
            data: POST 데이터

        Returns:
            API 응답 데이터
        """
        self._check_rate_limit()

        # 액세스 토큰 추가
        if params is None:
            params = {}
        params["access_token"] = self.access_token

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                timeout=30
            )

            # Rate Limit 헤더 파싱
            self._parse_rate_limit_headers(response.headers)

            # 응답 처리
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 3600))
                raise RateLimitError(
                    "Rate limit 초과. 나중에 다시 시도하세요.",
                    retry_after=retry_after
                )
            else:
                error_data = response.json().get("error", {})
                raise InstagramAPIError(
                    error_data.get("message", f"HTTP {response.status_code}"),
                    error_code=error_data.get("code"),
                    error_subcode=error_data.get("error_subcode")
                )

        except requests.exceptions.RequestException as e:
            raise InstagramAPIError(f"네트워크 오류: {str(e)}")

    def get_my_profile(self) -> Dict:
        """
        자신의 비즈니스 계정 프로필 조회

        Returns:
            프로필 정보
        """
        url = f"{INSTAGRAM_GRAPH_API_BASE_URL}/{self.business_account_id}"
        params = {"fields": ",".join(PROFILE_FIELDS)}

        return self._request("GET", url, params)

    def get_my_media(self, limit: int = 25) -> List[Dict]:
        """
        자신의 계정 미디어(게시물) 조회

        Args:
            limit: 가져올 게시물 수

        Returns:
            게시물 리스트
        """
        url = f"{INSTAGRAM_GRAPH_API_BASE_URL}/{self.business_account_id}/media"
        params = {
            "fields": ",".join(MEDIA_FIELDS),
            "limit": min(limit, 100)  # 최대 100개
        }

        response = self._request("GET", url, params)
        return response.get("data", [])

    def get_business_discovery(
        self,
        username: str,
        include_media: bool = True,
        media_limit: int = 25
    ) -> Optional[Dict]:
        """
        다른 비즈니스/크리에이터 계정 정보 조회 (Business Discovery API)

        주의: 대상 계정이 비즈니스 또는 크리에이터 계정이어야 합니다.

        Args:
            username: 조회할 Instagram 사용자명 (@제외)
            include_media: 미디어 포함 여부
            media_limit: 가져올 미디어 수

        Returns:
            계정 정보 또는 None (조회 실패 시)
        """
        # 캐시 확인
        cache_key = f"profile_{username}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached.get("_cached_at", 0) < 86400:  # 24시간 캐시
                return cached.get("data")

        # Business Discovery 필드 구성
        fields = ",".join(BUSINESS_DISCOVERY_FIELDS)

        if include_media:
            media_fields = ",".join(BUSINESS_DISCOVERY_MEDIA_FIELDS)
            fields += f",media.limit({media_limit}){{{media_fields}}}"

        url = f"{INSTAGRAM_GRAPH_API_BASE_URL}/{self.business_account_id}"
        params = {
            "fields": f"business_discovery.username({username}){{{fields}}}"
        }

        try:
            response = self._request("GET", url, params)
            result = response.get("business_discovery")

            # 캐시 저장
            if result:
                self._cache[cache_key] = {
                    "data": result,
                    "_cached_at": time.time()
                }

            return result

        except InstagramAPIError as e:
            if e.error_code == 100:  # 사용자를 찾을 수 없음 또는 비즈니스 계정이 아님
                logger.warning(f"사용자 '{username}'을(를) 찾을 수 없거나 비즈니스 계정이 아닙니다.")
                return None
            raise

    def get_hashtag_id(self, hashtag: str) -> Optional[str]:
        """
        해시태그 ID 조회

        Args:
            hashtag: 해시태그 이름 (#제외)

        Returns:
            해시태그 ID
        """
        # 주간 해시태그 검색 제한 체크
        week_ago = datetime.now() - timedelta(days=7)
        recent_searches = {
            tag: ts for tag, ts in self._hashtag_searches.items()
            if ts > week_ago
        }
        self._hashtag_searches = recent_searches

        # 이미 검색한 해시태그는 제한에 포함되지 않음
        if hashtag not in self._hashtag_searches:
            if len(self._hashtag_searches) >= HASHTAG_SEARCH_LIMIT_PER_WEEK:
                raise InstagramAPIError(
                    f"주간 해시태그 검색 제한({HASHTAG_SEARCH_LIMIT_PER_WEEK}개)에 도달했습니다."
                )
            self._hashtag_searches[hashtag] = datetime.now()

        url = f"{INSTAGRAM_GRAPH_API_BASE_URL}/ig_hashtag_search"
        params = {
            "user_id": self.business_account_id,
            "q": hashtag
        }

        response = self._request("GET", url, params)
        data = response.get("data", [])

        if data:
            return data[0].get("id")
        return None

    def search_hashtag(
        self,
        hashtag: str,
        search_type: str = "recent",
        limit: int = 50
    ) -> List[Dict]:
        """
        해시태그로 게시물 검색

        Args:
            hashtag: 해시태그 이름 (#제외)
            search_type: "recent" (최근) 또는 "top" (인기)
            limit: 가져올 게시물 수 (최대 50)

        Returns:
            게시물 리스트
        """
        hashtag_id = self.get_hashtag_id(hashtag)
        if not hashtag_id:
            return []

        endpoint = "recent_media" if search_type == "recent" else "top_media"
        url = f"{INSTAGRAM_GRAPH_API_BASE_URL}/{hashtag_id}/{endpoint}"

        # 해시태그 검색에서 사용 가능한 필드
        available_fields = [
            "id",
            "caption",
            "media_type",
            "timestamp",
            "like_count",
            "comments_count",
            "permalink"
        ]

        params = {
            "user_id": self.business_account_id,
            "fields": ",".join(available_fields),
            "limit": min(limit, 50)  # 최대 50개
        }

        response = self._request("GET", url, params)
        return response.get("data", [])

    def discover_influencers_by_hashtag(
        self,
        hashtags: List[str],
        min_likes: int = 100,
        limit_per_hashtag: int = 30
    ) -> List[Dict]:
        """
        해시태그로 인플루언서 발굴

        게시물에서 작성자를 추출하고 프로필 정보를 조회합니다.

        Args:
            hashtags: 검색할 해시태그 리스트
            min_likes: 최소 좋아요 수 필터
            limit_per_hashtag: 해시태그당 검색할 게시물 수

        Returns:
            발굴된 인플루언서 리스트
        """
        discovered = {}  # username -> profile data

        for hashtag in hashtags:
            try:
                posts = self.search_hashtag(hashtag, "top", limit_per_hashtag)

                for post in posts:
                    # 좋아요 필터
                    if post.get("like_count", 0) < min_likes:
                        continue

                    # 게시물에서 permalink를 통해 사용자명 추출
                    permalink = post.get("permalink", "")
                    if "/p/" in permalink:
                        # permalink 형식: https://www.instagram.com/p/ABC123/
                        # 사용자명은 직접 얻을 수 없으므로 별도 처리 필요
                        pass

                logger.info(f"해시태그 #{hashtag}에서 {len(posts)}개 게시물 발견")

            except InstagramAPIError as e:
                logger.error(f"해시태그 #{hashtag} 검색 실패: {e}")
                continue

        return list(discovered.values())

    def get_influencer_profile(self, username: str) -> Optional[Dict]:
        """
        인플루언서 프로필 정보를 현재 시스템 형식으로 변환하여 반환

        Args:
            username: Instagram 사용자명

        Returns:
            시스템 형식의 인플루언서 데이터
        """
        profile = self.get_business_discovery(username, include_media=True, media_limit=50)

        if not profile:
            return None

        # 최근 게시물 변환
        recent_posts = []
        media_data = profile.get("media", {}).get("data", [])

        for post in media_data[:20]:  # 최근 20개
            recent_posts.append({
                "caption": post.get("caption", ""),
                "likes": post.get("like_count", 0),
                "comments": post.get("comments_count", 0),
                "timestamp": post.get("timestamp", ""),
                "media_type": post.get("media_type", "IMAGE")
            })

        # 시스템 형식으로 변환
        return {
            "username": profile.get("username", username),
            "followers": profile.get("followers_count", 0),
            "following": profile.get("follows_count", 0),
            "bio": profile.get("biography", ""),
            "profile_picture_url": profile.get("profile_picture_url", ""),
            "website": profile.get("website", ""),
            "media_count": profile.get("media_count", 0),
            "recent_posts": recent_posts,
            "audience_countries": {},  # API에서 직접 제공하지 않음
            "avg_upload_interval_days": self._calculate_upload_interval(recent_posts),
            "_fetched_at": datetime.now().isoformat(),
            "_source": "instagram_api"
        }

    def _calculate_upload_interval(self, posts: List[Dict]) -> float:
        """게시물 업로드 간격 계산"""
        if len(posts) < 2:
            return 0.0

        timestamps = []
        for post in posts:
            ts = post.get("timestamp")
            if ts:
                try:
                    # ISO 형식 파싱
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    timestamps.append(dt)
                except (ValueError, AttributeError):
                    continue

        if len(timestamps) < 2:
            return 0.0

        timestamps.sort(reverse=True)
        intervals = []

        for i in range(len(timestamps) - 1):
            diff = (timestamps[i] - timestamps[i + 1]).days
            if diff > 0:
                intervals.append(diff)

        if intervals:
            return sum(intervals) / len(intervals)
        return 0.0

    def refresh_access_token(self) -> Dict:
        """
        장기 토큰 갱신

        Returns:
            새 토큰 정보
        """
        if not self.app_secret:
            raise InstagramAPIError("토큰 갱신을 위해 FACEBOOK_APP_SECRET이 필요합니다.")

        url = f"{INSTAGRAM_GRAPH_API_BASE_URL}/refresh_access_token"
        params = {
            "grant_type": "ig_refresh_token",
            "access_token": self.access_token
        }

        response = self._request("GET", url, params)

        if "access_token" in response:
            self.access_token = response["access_token"]
            logger.info(f"토큰 갱신 완료. 만료: {response.get('expires_in', 0)}초")

        return response

    def get_rate_limit_status(self) -> Dict:
        """현재 Rate Limit 상태 반환"""
        return {
            "call_count": self.rate_limit_status.call_count,
            "total_time": self.rate_limit_status.total_time,
            "percentage": self.rate_limit_status.percentage,
            "calls_remaining": max(0, RATE_LIMIT_PER_HOUR - len(self._call_timestamps)),
            "hashtags_remaining": max(0, HASHTAG_SEARCH_LIMIT_PER_WEEK - len(self._hashtag_searches))
        }


# 싱글톤 인스턴스 (환경변수 설정 시 자동 생성)
_api_instance: Optional[InstagramGraphAPI] = None


def get_instagram_api() -> InstagramGraphAPI:
    """Instagram API 싱글톤 인스턴스 반환"""
    global _api_instance

    if _api_instance is None:
        _api_instance = InstagramGraphAPI()

    return _api_instance


# 테스트 코드
if __name__ == "__main__":
    # 환경변수 설정 확인
    if not INSTAGRAM_ACCESS_TOKEN or not INSTAGRAM_BUSINESS_ACCOUNT_ID:
        print("환경변수를 설정해주세요:")
        print("  export INSTAGRAM_ACCESS_TOKEN='your_token'")
        print("  export INSTAGRAM_BUSINESS_ACCOUNT_ID='your_business_account_id'")
        exit(1)

    api = InstagramGraphAPI()

    # 자신의 프로필 조회
    print("\n=== 내 프로필 ===")
    my_profile = api.get_my_profile()
    print(json.dumps(my_profile, ensure_ascii=False, indent=2))

    # Rate Limit 상태
    print("\n=== Rate Limit 상태 ===")
    status = api.get_rate_limit_status()
    print(json.dumps(status, ensure_ascii=False, indent=2))