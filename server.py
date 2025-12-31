"""
AI 헤어 인플루언서 큐레이션 에이전트 - FastAPI 서버
==================================================

학술적 알고리즘 기반 인플루언서 추천 시스템

파이프라인:
1. 크롤링: 브랜드/인플루언서 데이터 수집 (Instagram Graph API)
2. 분석: FIS 허수 탐지 (Benford's Law + Chi-squared)
3. 분류: Expert/Trendsetter (TF-IDF + Cosine Similarity)
4. 인덱싱: ChromaDB 벡터 저장 + LLM 페르소나 생성
5. 검색: Hybrid Scoring (Vector + FIS + RRF)
6. 추천: Temperature Scaling + 상세 추천 사유

핵심 기술:
- RAG: ChromaDB + OpenAI Embeddings
- LLM: GPT-4o-mini (페르소나 생성)
- 알고리즘: Benford's Law, TF-IDF, RRF, Temperature Scaling

실행: python server.py
API 문서: http://localhost:8000/docs
"""

import os
import sys
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from api.routes import router, init_routes


# ============== 설정 ==============

HOST = "0.0.0.0"
PORT = 8000


# ============== 데이터 로드 ==============

def load_data():
    """데이터 로드"""
    # 인플루언서 데이터
    with open(os.path.join(BASE_DIR, "data", "influencers_data.json"), "r", encoding="utf-8") as f:
        influencers = json.load(f)["influencers"]

    # 브랜드 데이터
    with open(os.path.join(BASE_DIR, "data", "amore_brands.json"), "r", encoding="utf-8") as f:
        brand_db = json.load(f)

    return influencers, brand_db


# ============== 앱 생성 ==============

app = FastAPI(
    title="AI 헤어 인플루언서 큐레이션 에이전트",
    description="브랜드-인플루언서 최적 매칭을 위한 AI 기반 추천 시스템",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일
static_path = os.path.join(BASE_DIR, "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# 데이터 및 라우터 초기화
INFLUENCERS, BRAND_DB = load_data()
init_routes(BRAND_DB, INFLUENCERS)
app.include_router(router, prefix="/api")

# RAG 인덱싱 (서버 시작 시 자동 실행)
def init_rag_index():
    """RAG 인덱스 초기화 - 인덱싱이 안 되어 있으면 자동 실행"""
    try:
        from pipeline import RAG_AVAILABLE, InfluencerAnalysisManager
        if not RAG_AVAILABLE:
            print("RAG 시스템 비활성화 (ChromaDB 없음)")
            return

        manager = InfluencerAnalysisManager()
        stats = manager.rag.get_stats() if hasattr(manager.rag, 'get_stats') else {}
        indexed_count = len(manager.rag.get_all_usernames()) if hasattr(manager.rag, 'get_all_usernames') else 0

        if indexed_count == 0:
            print(f"RAG 인덱싱 시작 ({len(INFLUENCERS)}명)...")
            result = manager.analyze_and_index_all(INFLUENCERS)
            print(f"RAG 인덱싱 완료: {result.get('indexed', 0)}명 인덱싱됨")
        else:
            print(f"RAG 인덱스 로드됨: {indexed_count}명")
    except Exception as e:
        print(f"RAG 초기화 오류: {e}")

# 앱 시작 시 RAG 초기화
init_rag_index()


# ============== 메인 페이지 ==============

@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)

    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI 헤어 인플루언서 큐레이션</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            a { color: #0066cc; }
            .api-link { background: #f5f5f5; padding: 10px 20px; border-radius: 5px; display: inline-block; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>AI 헤어 인플루언서 큐레이션 에이전트</h1>
        <p>API 서버가 실행 중입니다.</p>
        <div class="api-link">
            <a href="/docs">API 문서 보기 (Swagger UI)</a>
        </div>
    </body>
    </html>
    """)


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "influencers_count": len(INFLUENCERS),
        "brands_count": len(BRAND_DB.get("brands", {}))
    }


# ============== 서버 실행 ==============

def main():
    """서버 실행"""
    import uvicorn

    print()
    print("=" * 50)
    print("  AI 헤어 인플루언서 큐레이션 에이전트")
    print("=" * 50)
    print(f"  서버 주소: http://localhost:{PORT}")
    print(f"  API 문서:  http://localhost:{PORT}/docs")
    print("  종료하려면 Ctrl+C를 누르세요")
    print("=" * 50)
    print()

    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
