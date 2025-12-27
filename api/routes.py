"""
API 라우터 - 모든 엔드포인트 정의
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging

from config.products import PRODUCT_CATEGORIES
from modules.taxonomy import analyze_influencer
from modules.fis_engine import calculate_fis_score
from modules.matcher import get_full_recommendations
from modules.rag_matcher import match_with_campaign

# Instagram 스크래퍼 (옵셔널)
try:
    from services.instagram_scraper import get_scraper_service
    _scraper_available = True
except ImportError:
    _scraper_available = False

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Pydantic 모델 ==============

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class BrandQuery(BaseModel):
    brand_name: str
    product_type: Optional[str] = None
    description: Optional[str] = None
    expert_count: Optional[int] = 2
    trendsetter_count: Optional[int] = 3


class CampaignQuery(BaseModel):
    """RAG 기반 캠페인 매칭용 쿼리"""
    brand_name: str
    campaign_query: str  # 자연어 캠페인 설명
    top_k: Optional[int] = 5
    min_fis: Optional[int] = 60


# ============== 의존성 (나중에 주입됨) ==============

_chatbot = None
_influencers = None
_brand_db = None


def init_routes(chatbot, influencers, brand_db):
    """라우터 초기화"""
    global _chatbot, _influencers, _brand_db
    _chatbot = chatbot
    _influencers = influencers
    _brand_db = brand_db


# ============== 챗봇 API ==============

@router.post("/chat")
async def chat(message: ChatMessage):
    """챗봇 대화 API"""
    session_id = message.session_id or "default"
    return _chatbot.process_message(message.message, session_id)


# ============== 브랜드 API ==============

@router.get("/brands")
async def get_brands():
    """브랜드 목록 조회"""
    hair_brands = _brand_db.get("hair_brands", [])
    brands_detail = {}

    for brand_name in hair_brands:
        brand_info = _brand_db.get("brands", {}).get(brand_name, {})
        brands_detail[brand_name] = {
            "slogan": brand_info.get("slogan", ""),
            "aesthetic_style": brand_info.get("aesthetic_style", ""),
            "core_values": brand_info.get("core_values", [])
        }

    return {
        "brands": list(_brand_db.get("brands", {}).keys()),
        "hair_brands": hair_brands,
        "brands_detail": brands_detail
    }


@router.get("/brands/{brand_name}")
async def get_brand_info(brand_name: str):
    """특정 브랜드 정보 조회"""
    brand_info = _brand_db.get("brands", {}).get(brand_name)
    if not brand_info:
        raise HTTPException(status_code=404, detail="브랜드를 찾을 수 없습니다")
    return brand_info


# ============== 추천 API ==============

@router.post("/recommend")
async def recommend_influencers(query: BrandQuery):
    """인플루언서 추천 API"""
    brand_info = _brand_db.get("brands", {}).get(query.brand_name, {})

    description = query.description or ""
    extracted_style = _chatbot.extract_aesthetic_style(description)
    extracted_audience = _chatbot.extract_target_audience(description)

    brand_data = {
        "brand_name": query.brand_name,
        "slogan": brand_info.get("slogan", ""),
        "core_values": brand_info.get("core_values", []),
        "target_audience": extracted_audience or brand_info.get("age_target", ""),
        "product_type": query.product_type or brand_info.get("product_categories", ["샴푸"])[0],
        "aesthetic_style": extracted_style or brand_info.get("aesthetic_style", "Trendy"),
        "campaign_description": description
    }

    expert_count = query.expert_count or 2
    trendsetter_count = query.trendsetter_count or 3
    total_count = expert_count + trendsetter_count

    results = get_full_recommendations(
        brand_data,
        _influencers,
        top_k=total_count,
        min_fis=60,
        expert_count=expert_count,
        trendsetter_count=trendsetter_count
    )

    results["brand_info"] = {
        "name": query.brand_name,
        "slogan": brand_info.get("slogan", ""),
        "aesthetic_style": brand_data["aesthetic_style"],
        "target_audience": brand_data["target_audience"],
        "product_type": brand_data["product_type"]
    }

    return results


@router.post("/recommend-campaign")
async def recommend_by_campaign(query: CampaignQuery):
    """
    RAG 기반 캠페인 맞춤 인플루언서 추천 API

    자연어로 캠페인을 설명하면 캠페인 유형, 제품 카테고리,
    타겟 오디언스를 자동으로 파악하여 최적의 인플루언서를 추천합니다.

    예시 쿼리:
    - "탈모 고민 있는 30대 남성 대상 루트젠 샴푸 캠페인"
    - "20대 여성 대상 염색 후 손상모 케어 제품 홍보"
    - "프리미엄 향수 라인 런칭, 세련된 이미지 필요"
    """
    brand_info = _brand_db.get("brands", {}).get(query.brand_name, {})

    if not brand_info:
        raise HTTPException(status_code=404, detail=f"브랜드 '{query.brand_name}'을(를) 찾을 수 없습니다")

    # 브랜드 데이터 구성
    brand_data = {
        "brand_name": query.brand_name,
        "slogan": brand_info.get("slogan", ""),
        "core_values": brand_info.get("core_values", []),
        "target_keywords": brand_info.get("target_keywords", []),
        "aesthetic_style": brand_info.get("aesthetic_style", "Trendy"),
        "product_categories": brand_info.get("product_categories", []),
        "product_lines": brand_info.get("product_lines", []),
        "price_tier": brand_info.get("price_tier", "Mass"),
        "age_target": brand_info.get("age_target", "")
    }

    try:
        # RAG 매처 호출
        results = match_with_campaign(
            campaign_query=query.campaign_query,
            brand_data=brand_data,
            influencers=_influencers,
            top_k=query.top_k or 5,
            min_fis=query.min_fis or 60
        )

        # 브랜드 정보 추가
        results["brand_info"] = {
            "name": query.brand_name,
            "slogan": brand_info.get("slogan", ""),
            "aesthetic_style": brand_data["aesthetic_style"],
            "price_tier": brand_data["price_tier"]
        }

        return results

    except Exception as e:
        logger.error(f"RAG 매칭 실패: {e}")
        raise HTTPException(status_code=500, detail=f"매칭 처리 중 오류 발생: {str(e)}")


# ============== 제품 API ==============

@router.get("/product-categories")
async def get_product_categories():
    """1단계: 제품 카테고리 목록"""
    categories = []
    for name, info in PRODUCT_CATEGORIES.items():
        categories.append({
            'name': name,
            'description': info['description'],
            'icon': info['icon'],
            'target': info['target'],
            'product_count': len(info['products'])
        })

    return {"categories": categories, "total_categories": len(categories)}


@router.get("/product-categories/{category_name}")
async def get_products_by_category(category_name: str):
    """2단계: 세부 제품 목록"""
    if category_name not in PRODUCT_CATEGORIES:
        raise HTTPException(status_code=404, detail=f"카테고리 '{category_name}'을(를) 찾을 수 없습니다")

    info = PRODUCT_CATEGORIES[category_name]
    return {
        "category": category_name,
        "description": info['description'],
        "icon": info['icon'],
        "target": info['target'],
        "products": info['products'],
        "product_count": len(info['products'])
    }


@router.get("/product-types")
async def get_product_types():
    """제품 유형 전체 목록"""
    categories = {name: info['products'] for name, info in PRODUCT_CATEGORIES.items()}
    all_products = []
    for info in PRODUCT_CATEGORIES.values():
        all_products.extend(info['products'])

    return {"categories": categories, "all_products": all_products, "total": len(all_products)}


# ============== 인플루언서 API ==============

@router.get("/influencers")
async def get_influencers():
    """인플루언서 목록"""
    return {"influencers": _influencers, "total": len(_influencers)}


@router.get("/influencers/{username}")
async def get_influencer_analysis(username: str):
    """특정 인플루언서 분석"""
    influencer = next((inf for inf in _influencers if inf["username"] == username), None)
    if not influencer:
        raise HTTPException(status_code=404, detail="인플루언서를 찾을 수 없습니다")

    return {
        "influencer": influencer,
        "taxonomy": analyze_influencer(influencer),
        "fis": calculate_fis_score(influencer)
    }


# ============== Instagram API 스크래퍼 ==============

class ScrapeRequest(BaseModel):
    username: str


class BatchScrapeRequest(BaseModel):
    usernames: List[str]
    skip_existing: Optional[bool] = True


class HashtagSearchRequest(BaseModel):
    hashtags: List[str]
    min_likes: Optional[int] = 500
    min_followers: Optional[int] = 10000
    limit_per_hashtag: Optional[int] = 30


@router.get("/instagram/status")
async def get_instagram_status():
    """
    Instagram API 상태 확인

    Returns:
        API 연결 상태 및 Rate Limit 정보
    """
    if not _scraper_available:
        return {
            "available": False,
            "message": "Instagram 스크래퍼 모듈이 설치되지 않았습니다."
        }

    try:
        scraper = get_scraper_service()
        stats = scraper.get_statistics()

        return {
            "available": scraper.is_api_available(),
            "statistics": stats,
            "message": "API 연결됨" if scraper.is_api_available() else "API 토큰이 설정되지 않았습니다."
        }
    except Exception as e:
        logger.error(f"Instagram 상태 확인 실패: {e}")
        return {
            "available": False,
            "message": str(e)
        }


@router.post("/instagram/scrape")
async def scrape_influencer(request: ScrapeRequest):
    """
    Instagram에서 인플루언서 프로필 수집

    Args:
        request: 사용자명이 포함된 요청

    Returns:
        수집된 인플루언서 데이터
    """
    if not _scraper_available:
        raise HTTPException(
            status_code=503,
            detail="Instagram 스크래퍼가 사용 불가합니다."
        )

    try:
        scraper = get_scraper_service()

        if not scraper.is_api_available():
            raise HTTPException(
                status_code=503,
                detail="Instagram API가 설정되지 않았습니다. 환경변수를 확인하세요."
            )

        profile = scraper.add_influencer(request.username)

        if profile:
            return {
                "success": True,
                "data": profile,
                "message": f"@{request.username} 프로필을 성공적으로 수집했습니다."
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"@{request.username}을(를) 찾을 수 없거나 비즈니스 계정이 아닙니다."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"스크래핑 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/instagram/scrape-batch")
async def scrape_batch(request: BatchScrapeRequest):
    """
    여러 인플루언서 일괄 수집

    Args:
        request: 사용자명 리스트

    Returns:
        수집 결과 요약
    """
    if not _scraper_available:
        raise HTTPException(
            status_code=503,
            detail="Instagram 스크래퍼가 사용 불가합니다."
        )

    try:
        scraper = get_scraper_service()

        if not scraper.is_api_available():
            raise HTTPException(
                status_code=503,
                detail="Instagram API가 설정되지 않았습니다."
            )

        collected, failed = scraper.scrape_multiple(
            request.usernames,
            skip_existing=request.skip_existing
        )

        # 수집된 데이터 저장
        if collected:
            data = scraper.load_existing_data()
            existing_usernames = {inf.get("username") for inf in data.get("influencers", [])}

            for profile in collected:
                if profile.get("username") not in existing_usernames:
                    data["influencers"].append(profile)

            scraper.save_data(data)

        return {
            "success": True,
            "collected_count": len(collected),
            "failed_count": len(failed),
            "failed_usernames": failed,
            "message": f"{len(collected)}명 수집 완료, {len(failed)}명 실패"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"일괄 스크래핑 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/instagram/search-hashtag")
async def search_by_hashtag(request: HashtagSearchRequest):
    """
    해시태그로 인플루언서 검색

    주의: 주당 30개 고유 해시태그 검색 제한이 있습니다.

    Args:
        request: 해시태그 리스트 및 필터 옵션

    Returns:
        검색된 게시물 및 인플루언서 정보
    """
    if not _scraper_available:
        raise HTTPException(
            status_code=503,
            detail="Instagram 스크래퍼가 사용 불가합니다."
        )

    try:
        scraper = get_scraper_service()

        if not scraper.is_api_available():
            raise HTTPException(
                status_code=503,
                detail="Instagram API가 설정되지 않았습니다."
            )

        influencers = scraper.discover_by_hashtags(
            hashtags=request.hashtags,
            min_likes=request.min_likes,
            min_followers=request.min_followers,
            limit_per_hashtag=request.limit_per_hashtag
        )

        return {
            "success": True,
            "discovered_count": len(influencers),
            "influencers": influencers,
            "searched_hashtags": request.hashtags
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"해시태그 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/instagram/update-all")
async def update_all_influencers(max_updates: int = 50):
    """
    기존 인플루언서 데이터 일괄 업데이트

    Args:
        max_updates: 최대 업데이트 수 (Rate limit 고려, 기본 50)

    Returns:
        업데이트 결과
    """
    if not _scraper_available:
        raise HTTPException(
            status_code=503,
            detail="Instagram 스크래퍼가 사용 불가합니다."
        )

    try:
        scraper = get_scraper_service()

        if not scraper.is_api_available():
            raise HTTPException(
                status_code=503,
                detail="Instagram API가 설정되지 않았습니다."
            )

        success, fail = scraper.update_all_influencers(max_updates=max_updates)

        return {
            "success": True,
            "updated_count": success,
            "failed_count": fail,
            "message": f"{success}명 업데이트 완료, {fail}명 실패"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"일괄 업데이트 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/instagram/influencer/{username}")
async def remove_influencer(username: str):
    """
    인플루언서 삭제

    Args:
        username: 삭제할 사용자명

    Returns:
        삭제 결과
    """
    if not _scraper_available:
        raise HTTPException(
            status_code=503,
            detail="Instagram 스크래퍼가 사용 불가합니다."
        )

    try:
        scraper = get_scraper_service()
        removed = scraper.remove_influencer(username)

        if removed:
            return {
                "success": True,
                "message": f"@{username}이(가) 삭제되었습니다."
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"@{username}을(를) 찾을 수 없습니다."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))