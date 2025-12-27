#!/bin/bash
# AI 헤어 인플루언서 큐레이션 에이전트 - 실행 스크립트

cd "$(dirname "$0")"

echo ""
echo "=========================================="
echo "  AI 헤어 인플루언서 큐레이션 에이전트"
echo "=========================================="
echo ""

# 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "  [OK] 가상환경 활성화됨"
fi

# 서버 실행
echo "  [..] 서버 시작 중..."
echo ""
python server.py