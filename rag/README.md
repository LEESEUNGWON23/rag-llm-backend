# api-rag — RAG Pipeline

[← 메인으로](../README.md)

> **팀 구성:** Python 백엔드 개발자 2명 (본인 + 시니어 개발자). 아래 항목은 본인이 주도적으로 설계·구현한 내용입니다.

---

## 3. Key Technical Challenges & Solutions

### 3-1. Reranker 선택: 로컬 모델 비교 실험 → vLLM 원격 API 전환

**Situation**  
초기 구현에서는 `sentence_transformers`의 `CrossEncoder`를 서버에 직접 로드했습니다. 모델이 GPU 메모리를 상시 점유하고, 서버 재시작마다 수십 초의 모델 로딩 시간이 발생했습니다.

**Task**  
GPU 메모리 비점유, 빠른 추론 속도, 높은 reranking 정확도를 동시에 만족하는 reranker를 선택해야 했습니다.

**Action**  
세 가지 방식을 직접 비교 실험했습니다.

| | CrossEncoder (로컬 GPU) | FlashRank MultiBERT (로컬 CPU) | vLLM BGE-Reranker-v2-m3 (원격 API) |
|---|---|---|---|
| **추론 속도** | 측정 불가 (로딩 병목) | 약 5.4초 | **약 0.3초** |
| **정확도** | 정성 평가 | pairwise ranking (정성) | **score 0.86** (정량) |
| **인프라** | 로컬 GPU 점유 | 로컬 CPU 부하 | 외부 GPU 서버 API |

FlashRank는 GPU 없이 동작하지만 문서 수에 비례해 지연이 커졌습니다. vLLM은 정량적 relevance score를 반환해 threshold 기반 필터링도 가능했습니다.

**Result**  
vLLM BGE-Reranker-v2-m3으로 최종 전환 후, FlashRank 대비 **18배 속도 단축(5.4s → 0.3s)** 을 달성하고 로컬 서버 리소스를 완전히 해방했습니다.

```python
# Before: CrossEncoder 로컬 로드
from sentence_transformers import CrossEncoder
self.model = CrossEncoder(self.model_name, device="cuda")
scores = self.model.predict([[query, hit.text] for hit in hits])

# After: vLLM 원격 API 호출
response = requests.post(self.api_url, json={
    "model": self.model_name,
    "query": query,
    "documents": [hit.text for hit in hits],
    "top_n": len(hits),
})
```

---

### 3-2. Reranking Timeout: 단일 동기 요청 → 비동기 batch 병렬 처리

**Situation**  
vLLM API로 전환한 뒤, 검색 결과가 100건 이상일 때 모든 문서를 하나의 동기 요청에 담아 전송하면 timeout(10s)을 초과하는 경우가 발생했습니다.

**Action**  
문서를 batch로 나누고 `asyncio.gather`로 vLLM 서버에 **동시 전송**하는 방식으로 전환했습니다. 각 batch의 로컬 index에 `start_idx`를 더해 전체 hit 리스트 기준 index로 복원하는 것이 핵심이었습니다.

```python
# Before: 전체 문서를 하나의 동기 요청으로 전송
response = requests.post(self.api_url, json=payload, timeout=10)

# After: batch 분할 + asyncio.gather 동시 전송
async def _fetch_rerank_batch(self, session, query, batch_hits, start_idx) -> dict:
    async with session.post(self.api_url, json=payload, timeout=20) as resp:
        result = await resp.json()
        return {(item["index"] + start_idx): item["relevance_score"]
                for item in result.get("results", [])}

async def _run():
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                self._fetch_rerank_batch(session, query, hits[i:i+BATCH_SIZE], i)
            )
            for i in range(0, len(hits), BATCH_SIZE)
        ]
        return await asyncio.gather(*tasks)
```

**Result**  
Timeout 에러가 해소되었고, API 오류 시 unranked `hits[:top_k]`를 반환하는 fallback으로 서비스 중단을 방지했습니다.

---

### 3-3. Document Parser 교체: PaddleOCR → LangChain + Docling

**Situation**  
초기에는 PaddleOCR로 PDF와 이미지 문서를 처리했습니다. PaddleOCR은 **페이지 전체를 이미지로 변환한 뒤 OCR을 수행**하는 방식이라, 텍스트 PDF에서도 불필요하게 이미지화 과정을 거쳐 처리 속도가 느렸고 표·캡션 등 구조 정보를 잃었습니다.

**Action**  
파일 특성에 따라 parser를 분리했습니다.

- **LangChain loaders**: 텍스트 기반 문서(DOCX, PPTX, XLSX, CSV, 소스코드 등)는 포맷별 loader로 직접 파싱. 소스코드는 `LanguageParser`(tree-sitter 기반 AST chunking)로 함수·클래스 단위 분리.
- **OpenDataLoaderPDF**: 일반 PDF는 텍스트 레이어를 직접 추출. 이미지는 로컬 저장 후 스토리지에 업로드하고 URL을 payload에 연결.
- **Docling**: Figma export, 슬라이드 PDF 등 이미지 중심 문서에 한정 적용. bbox, 표, 캡션 구조를 보존.

```python
loader_map = {
    "pdf":  lambda: OpenDataLoaderPDFLoader(str(path), format="json", split_pages=True),
    "docx": lambda: UnstructuredWordDocumentLoader(str(path)),
    "pptx": lambda: UnstructuredPowerPointLoader(str(path)),
    "xlsx": lambda: UnstructuredExcelLoader(str(path)),
    # ...
}
# 소스코드: AST 기반 chunking
if ext in CODE_EXTENSIONS:
    parser = LanguageParser(language=EXTENSION_MAP[ext], parser_threshold=0)
    return GenericLoader.from_filesystem(path.parent, glob=path.name, parser=parser).load()
```

**Result**  
텍스트 PDF parsing 속도가 크게 향상되었고, 표 구조와 이미지 URL을 보존해 이후 citation 기능 구현의 기반이 되었습니다.

---

### 3-4. Citation 부정확: 임의 chunking → 페이지 단위 정렬

**Situation**  
LangChain text splitter가 token 크기 기준으로 chunk를 나누다 보니, **하나의 chunk가 페이지 경계를 넘어 2개 이상 페이지에 걸치는 경우**가 발생했습니다. citation 기능에서 페이지 번호가 부정확해지는 문제가 있었습니다.

**Action**  
`split_pages=True` 옵션으로 페이지 단위로 분리하고, chunk 수가 페이지 수와 일치하도록 강제했습니다. 각 chunk의 `page` 메타데이터를 payload에 저장해 검색 결과에 페이지 번호를 정확히 매핑했습니다.

**Result**  
Chunk-페이지 매핑이 정확해져 citation 시 "파일명 N페이지"를 신뢰성 있게 반환할 수 있게 되었습니다.

---

### 3-5. 알려진 한계: 특정 폰트 유니코드 깨짐

일부 PDF에서 텍스트를 추출하면 특정 문자가 '□', '?'로 깨지는 현상이 발생했습니다. 해당 PDF들이 사용한 폰트가 parsing 라이브러리에서 지원하지 않는 비표준 인코딩을 포함하고 있었으며, 유니코드가 깨지는 페이지를 감지하는 함수와 별도 처리 로직을 구현했으나 **라이브러리 레벨에서 폰트 자체를 지원하지 않는 경우 근본 해결이 불가능**했습니다. 완전한 해결을 위해서는 커스텀 폰트 매핑 또는 OCR fallback pipeline 구축이 필요합니다.

---

## 4. Performance Optimization

### Hybrid Search encoding 병렬화

Dense·Sparse encoding을 순차 실행하면 각 vLLM API 응답을 기다리는 시간이 두 배로 늘어납니다. 두 encoding은 서로 독립적이므로 `ThreadPoolExecutor`로 병렬 실행했습니다.

```python
# Before
dense_vector    = self.dense_model.encode(query_text)
sparse_vec_dict = self.sparse_model.encode(query_text)

# After
with ThreadPoolExecutor(max_workers=2) as executor:
    future_dense  = executor.submit(self.dense_model.encode, query_text)
    future_sparse = executor.submit(self.sparse_model.encode, query_text)
    dense_vector    = future_dense.result()
    sparse_vec_dict = future_sparse.result()
```

### RRF 선택 이유

Dense score(0~1)와 Sparse score(0~수십)는 scale이 달라 직접 합산하면 한쪽이 지배합니다. 가중치 수동 튜닝 없이 안정적으로 병합하기 위해 **순위 기반 RRF(Reciprocal Rank Fusion)** 를 선택했습니다.

### Embedding cache 도입

동일 텍스트가 반복 embedding되는 경우(재업로드, 부분 수정)를 줄이기 위해 embedding engine에 cache layer를 추가했습니다.

### embed_file batch 병렬화

문서 chunk embedding 시 chunk를 batch로 묶어 vLLM에 한 번에 요청하고, batch 간 병렬 처리를 적용해 대용량 문서 수집 시간을 단축했습니다.
