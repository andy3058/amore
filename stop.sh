#!/bin/bash
# AI 헤어 인플루언서 큐레이션 에이전트 - 종료 스크립트

echo ""
echo "서버 종료 중..."

# 포트 8000에서 실행 중인 프로세스 찾아서 종료
PID=$(lsof -ti:8000)

if [ -z "$PID" ]; then
    echo "  실행 중인 서버가 없습니다."
else
    kill -9 $PID
    echo "  [OK] 서버가 종료되었습니다. (PID: $PID)"
fi

echo ""