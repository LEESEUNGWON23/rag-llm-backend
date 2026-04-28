# RAG 기반 기업용 AI 어시스턴트 — 포트폴리오

> 기밀 유지를 위해 회사명 및 제품명은 표기하지 않습니다.

**기간:** 2026년 3월 ~ 4월 (약 2개월)  
**역할:** AI 백엔드 엔지니어 — RAG pipeline 및 LLM serving 백엔드 단독 설계·구현·배포  
**팀 구성:** Python 백엔드 개발자 2명 (본인 + 시니어 개발자), Spring 팀 별도

---

## 1. Project Overview

사내 문서를 기반으로 질문에 답변하는 **기업용 AI 어시스턴트** 서비스의 AI 백엔드를 담당했습니다.

사용자가 드라이브에 업로드한 문서(PDF, DOCX, PPTX, XLSX 등 15가지 이상 포맷)를 자동으로 파싱·embedding하고, 사용자 query가 들어오면 hybrid vector search → reranking → LLM 답변 생성 pipeline을 통해 출처가 명시된 답변을 streaming으로 제공합니다.

두 개의 서비스로 구성됩니다:
- **[api-rag →](rag/README.md)** : 문서 파싱, embedding, hybrid search, reranking
- **[be-fastapi →](fastapi/README.md)** : 채팅 API, LLM routing, 대화 context 관리

---

## 2. Architecture & Tech Stack

```
[ 문서 업로드 (15+ 포맷) ]
         │
         ▼
  [ 파일 파싱 ]
  LangChain Loaders ── 텍스트·코드·오피스 문서
  Docling           ── 이미지 중심 PDF, Figma export
         │
         ▼
  [ Embedding ]  Dense + Sparse 동시 생성 (vLLM 원격 serving)
         │
         ▼
  [ Qdrant 벡터 DB ]  prd / dev collection 분리
         │
         ▼
  [ Hybrid Search ]  Dense + Sparse → RRF fusion
         │
         ▼
  [ Reranker ]  vLLM BGE-Reranker-v2-m3 (비동기 batch)
         │
         ▼
  [ LLM 답변 생성 ]  SSE streaming / context 압축 / user memory
```

| 기술 | 선택 이유 |
|------|-----------|
| **Qdrant** | Dense + Sparse vector를 단일 collection에서 관리, RRF fusion query 네이티브 지원 |
| **vLLM (embedding·reranking)** | 로컬 GPU 점유 없이 원격 serving, FlashRank 대비 reranking 속도 18배 향상 |
| **LangChain** | 15가지 이상 포맷을 단일 인터페이스로 통합, AST 기반 코드 chunking 지원 |
| **Docling** | PaddleOCR(전체 페이지 이미지화)보다 빠르고 표·캡션 등 구조 정보 보존 |
| **FastAPI + asyncio** | SSE streaming과 비동기 LLM 호출을 단일 이벤트 루프에서 처리 |

---

## 3. 담당 범위

두 서비스에서 기능 단위로 과제를 받아 기술 조사 후 구현했습니다.

> **팀 구성:** Python 백엔드 개발자 2명 (본인 + 시니어 개발자), Spring 팀 별도. 아래 항목은 본인이 주도적으로 설계·구현한 내용입니다.

### RAG Pipeline

| 담당 기능 | 내용 |
|----------|------|
| Reranker 기술 조사 및 적용 | vLLM vs FlashRank 직접 비교 실험 → 추론 속도 **18배** 차이 확인, vLLM 채택 |
| MMR Selector 성능 개선 | 개별 encode 호출 → 배치 1회 호출로 변경 |
| PDF 처리 파이프라인 기여 | 텍스트·스캔·복합 PDF 분기 처리 로직에 기능 추가 |
| 멀티 포맷 문서 처리 | 15가지 이상 포맷별 추출 방식 조사 및 구현 |
| Web Search 파이프라인 | DuckDuckGo 검색 + Trafilatura 병렬 스크래핑 구현 |
| LangChain 파일 처리 도입 | 코드·마크업·데이터 파일 처리에 LangChain 로더 적용 |
| 성능 최적화 | LRU 캐시, 병렬 인코딩, 비동기 배치 reranking 도입 |

→ 상세 내용: [rag/README.md](rag/README.md)

### Chat Backend

| 담당 기능 | 내용 |
|----------|------|
| Context Compression | 75% 토큰 임계값 기반 LLM 압축 로직 구현 |
| Citation 시스템 | 소스 인용 마커 구현 및 ID 버그 수정 |
| User Memory 자동 추출 | 대화에서 사용자 성향 추출 및 관련성 기반 선택 로직 구현 |
| Web Search context 통합 | 검색 결과를 SSE stream에 실시간 주입 |
| 멀티 모델 라우터 | GPT-nano / Qwen / GPT-OSS 단일 인터페이스로 통합 |

→ 상세 내용: [fastapi/README.md](fastapi/README.md)
