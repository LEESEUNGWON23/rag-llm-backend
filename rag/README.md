# api-rag — RAG 파이프라인

[← 메인으로](../README.md)

문서 수집 → 임베딩 → 하이브리드 검색 → 재랭킹 파이프라인을 단독으로 설계·구현했습니다.  
아래는 개발 과정에서 실제로 문제를 인식하고 개선한 사례들입니다.

---

## 1. 재랭커: 로컬 GPU 모델 → vLLM 원격 API

**문제:** `sentence_transformers`의 `CrossEncoder`를 서버 로컬에서 로드하는 구조였습니다.  
모델이 GPU 메모리를 상시 점유하고, 서버 재시작마다 모델 로딩 시간이 발생했습니다.

**변경 전**
```python
# sentence_transformers + torch 로컬 로드
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = settings.RERANK_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(self, query, hits, top_k=100, batch_size=16):
        pairs = [[query, hit.text] for hit in hits]
        scores = self.model.predict(pairs, batch_size=batch_size)
        for hit, score in zip(hits, scores):
            hit.rerank_score = float(score)
        hits.sort(key=lambda h: h.rerank_score, reverse=True)
        return hits[:top_k]
```

**변경 후**
```python
# vLLM 원격 API 호출로 전환, 로컬 GPU 비점유
import requests

class Reranker:
    def __init__(self, model_name: str = settings.RERANK_MODEL):
        self.api_url = settings.VLLM_RERANK_URL.rstrip("/") + "/rerank"
        self._warmup()

    def rerank(self, query, hits, top_k=60):
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [hit.text for hit in hits],
            "top_n": len(hits),
        }
        response = requests.post(self.api_url, json=payload, timeout=10)
        score_map = {item["index"]: item["relevance_score"]
                     for item in response.json().get("results", [])}
        for i, hit in enumerate(hits):
            hit.rerank_score = score_map.get(i, -10.0)
        hits.sort(key=lambda h: h.rerank_score, reverse=True)
        return hits[:top_k]
```

---

## 2. 재랭커: 단일 동기 요청 → 비동기 배치 병렬 처리

**문제:** 검색 결과가 많을수록 하나의 요청에 모든 문서를 담아 전송하다 보니,  
문서 수에 비례해 vLLM 응답 시간이 늘어나 타임아웃이 발생하는 경우가 생겼습니다.

**변경 전**
```python
# 모든 문서를 하나의 동기 요청으로 전송
response = requests.post(self.api_url, json=payload, timeout=10)
```

**변경 후**
```python
# 배치로 나눠 asyncio.gather로 동시 전송
import asyncio, aiohttp

async def _fetch_rerank_batch(self, session, query, batch_hits, start_idx) -> dict:
    payload = {
        "model": self.model_name,
        "query": query,
        "documents": [self._truncate_text_for_rerank(h.text) for h in batch_hits],
        "top_n": len(batch_hits),
    }
    async with session.post(self.api_url, json=payload, timeout=20) as resp:
        result = await resp.json()
        return {(item["index"] + start_idx): item["relevance_score"]
                for item in result.get("results", [])}

def rerank(self, query, hits, top_k=60):
    async def _run():
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(
                    self._fetch_rerank_batch(session, query, hits[i:i+BATCH_SIZE], i)
                )
                for i in range(0, len(hits), BATCH_SIZE)
            ]
            results = await asyncio.gather(*tasks)
        score_map = {}
        for r in results:
            score_map.update(r)
        return score_map

    score_map = asyncio.run(_run())
    ...
```

각 배치의 로컬 인덱스에 `start_idx`를 더해 전체 hit 리스트 기준으로 복원하는 것이 핵심입니다.  
API 오류 시엔 재랭킹 없이 `hits[:top_k]`를 그대로 반환해 서비스 중단을 방지합니다.

---

## 3. 하이브리드 검색: 인코딩 순차 실행 → 병렬 실행

**문제:** Dense 벡터와 Sparse 벡터를 순차적으로 인코딩하고 있었습니다.  
두 모델은 서로 독립적인데 굳이 기다릴 이유가 없었습니다.

**변경 전**
```python
query_text = f"query: {text}"
dense_vector = self.dense_model.encode(query_text)       # 완료 후
sparse_vec_dict = self.sparse_model.encode(query_text)   # 실행
```

**변경 후**
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    future_dense  = executor.submit(self.dense_model.encode, query_text)
    future_sparse = executor.submit(self.sparse_model.encode, query_text)
    dense_vector     = future_dense.result()
    sparse_vec_dict  = future_sparse.result()
```

검색 결과 병합은 가중 합산 대신 **RRF(Reciprocal Rank Fusion)** 를 사용합니다.  
두 벡터 공간의 점수 스케일이 달라 직접 합산하면 보정이 필요하지만, RRF는 순위 기반이라 스케일 조정 없이 안정적으로 병합됩니다.

---

## 4. 문서 파싱: 포맷별 단일 처리 → 15가지 포맷 통합 파이프라인

**문제:** 파일 포맷마다 처리 로직이 분산되어 있어 새 포맷 추가 시 코드 여러 곳을 수정해야 했습니다.

**변경 후** — 단일 진입점에서 포맷별로 디스패치

```python
loader_map = {
    "pdf":  lambda: OpenDataLoaderPDFLoader(str(path), format="json", split_pages=True),
    "docx": lambda: UnstructuredWordDocumentLoader(str(path)),
    "pptx": lambda: UnstructuredPowerPointLoader(str(path)),
    "xlsx": lambda: UnstructuredExcelLoader(str(path)),
    "csv":  lambda: CSVLoader(str(path), encoding="utf-8"),
    "html": lambda: BSHTMLLoader(str(path)),
    # ...
}

# 소스코드: AST 기반 LanguageParser (Python, JS, Go, Rust 등 10개 언어)
if ext in settings.CODE_EXTENSIONS:
    parser = LanguageParser(language=EXTENSION_MAP.get(ext), parser_threshold=0)
    loader = GenericLoader.from_filesystem(path.parent, glob=path.name, parser=parser)
    return loader.load()

# JSON: 유효성 검사 → JSONL 폴백 → TextLoader 폴백 순서로 처리
if ext == "json":
    try:
        validate_json_file(path)
        return load_json_or_jsonl(path)
    except ValueError:
        return TextLoader(str(path), encoding="utf-8").load()
```

PDF 이미지는 추출 후 파일 스토리지에 업로드하고 URL을 페이로드에 연결합니다.  
Docling을 사용하는 이미지 중심 문서(Figma 내보내기 등)는 URL 연결을 **백그라운드 작업**으로 분리해 수집 지연을 없앴습니다.

---

## 5. 테이블 파싱: 구조 무시 → span-aware 정규화

**문제:** PDF/DOCX에서 추출된 테이블 페이로드에 `row_span` / `col_span`이 포함되어 있었는데,  
이를 무시하고 그대로 LLM에 넘기면 셀 위치가 어긋나 LLM이 잘못된 컬럼으로 읽는 문제가 발생했습니다.

**변경 후** — 2D 그리드로 재구성, span 전파 처리

```python
grid     = [[""] * total_cols for _ in range(total_rows)]
occupied = [[False] * total_cols for _ in range(total_rows)]

for cell in cells:
    r0, c0 = cell["row"] - 1, cell["col"] - 1
    rs, cs = cell.get("row_span", 1), cell.get("col_span", 1)
    if occupied[r0][c0]:
        continue
    for rr in range(r0, r0 + rs):
        for cc in range(c0, c0 + cs):
            grid[rr][cc] = cell_text
            occupied[rr][cc] = True

# 추출기가 row_span을 생략한 행: 위 행 값을 아래로 전파해 보정
for c in range(total_cols):
    last = ""
    for r in range(len(grid)):
        grid[r][c] = grid[r][c] or last
        if grid[r][c]: last = grid[r][c]
```

재구성된 그리드는 Markdown-KV 레코드 형식으로 직렬화하고,  
가장 가까운 heading/caption을 `context:` 필드로 각 행에 주입해 LLM이 섹션 맥락을 파악할 수 있게 했습니다.
