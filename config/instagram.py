"""
Instagram Graph API 설정
"""
import os
from typing import Optional

# API 기본 설정
INSTAGRAM_API_VERSION = "v21.0"
INSTAGRAM_GRAPH_API_BASE_URL = f"https://graph.instagram.com/{INSTAGRAM_API_VERSION}"
FACEBOOK_GRAPH_API_BASE_URL = f"https://graph.facebook.com/{INSTAGRAM_API_VERSION}"

# 환경 변수에서 토큰 로드
INSTAGRAM_ACCESS_TOKEN: Optional[str] = os.getenv("INSTAGRAM_ACCESS_TOKEN")
INSTAGRAM_BUSINESS_ACCOUNT_ID: Optional[str] = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID")
FACEBOOK_APP_SECRET: Optional[str] = os.getenv("FACEBOOK_APP_SECRET")

# API 필드 설정
PROFILE_FIELDS = [
    "username",
    "name",
    "biography",
    "followers_count",
    "follows_count",
    "media_count",
    "profile_picture_url",
    "website",
    "ig_id"
]

MEDIA_FIELDS = [
    "id",
    "caption",
    "media_type",
    "media_url",
    "permalink",
    "timestamp",
    "like_count",
    "comments_count",
    "thumbnail_url"
]

# Business Discovery로 다른 계정 조회 시 사용 가능한 필드
BUSINESS_DISCOVERY_FIELDS = [
    "username",
    "name",
    "biography",
    "followers_count",
    "follows_count",
    "media_count",
    "profile_picture_url",
    "website"
]

BUSINESS_DISCOVERY_MEDIA_FIELDS = [
    "id",
    "caption",
    "media_type",
    "timestamp",
    "like_count",
    "comments_count"
]

# Rate Limit 설정
RATE_LIMIT_PER_HOUR = 200  # 2025년 기준 시간당 200 호출
HASHTAG_SEARCH_LIMIT_PER_WEEK = 30  # 주당 30개 고유 해시태그

# 캐시 설정 (초 단위)
PROFILE_CACHE_TTL = 86400 * 7  # 7일
MEDIA_CACHE_TTL = 86400  # 1일
HASHTAG_CACHE_TTL = 86400  # 1일

# 헤어 관련 해시태그 목록
HAIR_HASHTAGS = [
    "헤어스타일",
    "헤어스타일링",
    "헤어케어",
    "염색",
    "펌",
    "헤어컬러",
    "뷰티",
    "미용사",
    "헤어디자이너",
    "살롱",
    "haircare",
    "hairstylist",
    "haircolor",
    "kbeauty"
]