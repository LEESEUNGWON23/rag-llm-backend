# be-fastapi — LLM 채팅 백엔드

[← 메인으로](../README.md)

> **팀 구성:** Python 백엔드 개발자 2명 (본인 + 시니어 개발자). 아래 항목은 본인이 주도적으로 설계·구현한 내용입니다.

---

## 3. Key Technical Challenges & Solutions

### 3-1. Context 압축: 정보 유실 → 구조화 prompt + 대용량 병렬 처리

**Situation**  
대화가 길어질수록 이전 대화를 모두 prompt에 넣으면 모델 최대 input token을 초과합니다. 초기 구현은 단순한 한 줄 요약 지시문으로 LLM에게 압축을 맡겼으나, **고유명사·수치·기술 용어가 압축 과정에서 유실**되어 후속 대화에서 LLM이 맥락을 잃는 문제가 발생했습니다.

**Action — 1단계: Prompt 구조화**  
역할·보존 대상·출력 형식을 명시적으로 지정하는 system prompt로 교체했습니다.

```python
# Before: 단순 한 줄 요약 지시
final_prompt = (
    "Please analyze and summarize the following information concisely. "
    f"Remove any redundant or unnecessary details:\n\n{combined_content}"
)
model = settings.GPTOSS_MODEL  # 모델 하드코딩

# After: 역할·보존 대상·출력 형식 명시
CONTEXT_COMPRESSION_SYSTEM_PROMPT = """
## Role
You are a Context Optimizer. Compress conversation history while maintaining 100% of essential context.

## Instructions
1. Extract Core Intent: user's primary goal and assistant's key answers.
2. Preserve Entities: names, dates, numbers, technical terms, unique identifiers.
3. Remove Noise: greetings, fillers, redundant explanations.
4. Format: dense, telegraphic style ("User asked X; Assistant provided Y using Z").
5. Target length: {target_len} characters.
"""

def compress_context(combined_content: str, model: str = None) -> str:
    target_len = len(combined_content) // 4  # target: 원본의 25%
    ...
```

**Action — 2단계: 대용량 context 처리**  
Context가 LLM 최대 input(100,000자)을 초과하면 압축 자체가 실패합니다. chunk로 분할하고 `ThreadPoolExecutor`로 병렬 압축한 뒤 합산하도록 개선했습니다.

```python
CHUNK_SIZE = 100_000
if len(combined_content) > CHUNK_SIZE:
    chunks = [combined_content[i:i + CHUNK_SIZE]
              for i in range(0, len(combined_content), CHUNK_SIZE)]
    with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as executor:
        compressed_chunks = list(executor.map(
            lambda c: compress_context(c, model=model), chunks
        ))
    return "\n".join(compressed_chunks)
```

**Result**  
정보 유실 없이 원본의 25% 수준으로 압축되었고, 대용량 context에서도 압축 실패 없이 처리됩니다. LLM 오류 시 원본 context를 그대로 반환하는 fallback으로 데이터 유실을 방지했습니다.

---

### 3-2. 클라이언트 연결 끊김 시 대화 유실 → asyncio.shield로 저장 보호

**Situation**  
SSE streaming 도중 클라이언트가 연결을 끊으면 `asyncio.CancelledError`가 발생합니다. `finally` 블록에서 대화 이력을 저장하는 `_finalize_response`를 호출하고 있었지만, **CancelledError가 `await` 중인 task까지 취소**시켜 저장이 중간에 중단되는 경우가 생겼습니다.

**Action**  
`asyncio.shield`로 저장 task를 외부 CancelledError로부터 격리했습니다. 저장 완료 후에 CancelledError를 다시 raise해 정상적인 취소 흐름도 유지했습니다.

```python
# Before: CancelledError 발생 시 저장 중단 가능
finally:
    await _finalize_response(...)

# After: asyncio.shield로 저장 task를 취소로부터 보호
finally:
    finalize_task = asyncio.create_task(_finalize_response(...))
    await asyncio.shield(finalize_task)  # 외부 CancelledError와 무관하게 완료 보장

    if cancelled:
        raise asyncio.CancelledError  # 저장 완료 후 취소 흐름 복원
```

**Result**  
클라이언트가 streaming 중 연결을 끊어도 대화 이력과 user memory 업데이트가 누락 없이 저장됩니다.

---

### 3-3. Token 초과로 인한 응답 실패 → 실시간 긴급 압축

**Situation**  
RAG context, 웹 검색 결과, 대화 이력을 모두 합산한 prompt가 모델 최대 input token을 초과해 LLM이 응답을 거부하는 경우가 있었습니다.

**Action**  
Prompt 조립 후 token 수를 체크하고, 모델 최대 input의 **75%** 를 초과하면 대화 이력만 즉시 압축 후 prompt를 재조립합니다. 75%를 기준으로 삼은 이유는 RAG context와 웹 검색 결과가 추가로 token을 차지하는 여유분을 남겨두기 위해서입니다.

```python
prompt_token_count = await asyncio.to_thread(get_tokenize, model=model, prompt=prompt)

if prompt_token_count > model_max_input_token * 0.75:
    conversation_context = await asyncio.to_thread(
        compress_context, conversation_context, model=model
    )
    prompt = DEFAULT_LLM_PROMPT.format(
        ..., conversation_context=conversation_context, ...
    )
```

**Result**  
Token 초과로 인한 LLM 응답 실패가 해소되었습니다. 압축은 대화 이력에만 적용하고 RAG·웹 context는 유지해 답변 품질 저하를 최소화했습니다.

---

### 3-4. 개인화 부재 → 대화 기반 User Memory 시스템 구현

**Situation**  
매 대화가 새로 시작되면 LLM은 사용자에 대한 정보가 전혀 없었습니다. 사용자가 자신의 직무, 선호 답변 방식, 반복적으로 묻는 맥락을 매번 다시 설명해야 했고 답변 품질이 일관되지 않았습니다.

**Action**  
10번째 turn마다 LLM이 대화에서 사용자 정보를 JSON 형태로 추출해 누적 저장하고, 다음 대화 시 현재 query와 관련성이 높은 memory만 선별해 prompt에 주입했습니다.

```python
# 대화에서 사용자 정보 추출 → 기존 memory와 중복 제거 후 병합
def update_user_context(new_dialogue, existing_user_context, model=None):
    new_items = json.loads(get_llm_content(...))  # LLM이 JSON list로 추출

    merged_texts = {m["text"] for m in existing_user_context}
    for item in new_items:
        if item.get("text") not in merged_texts:
            existing_user_context.append(item)
            merged_texts.add(item["text"])

    return existing_user_context[-50:]  # 최대 50개 유지


# Query와 관련성 높은 memory만 선별
def select_relevant_memories(query, memories, top_k=5):
    q_words = set(query.lower().split())
    scored = []
    always_include = []  # response_style은 항상 포함

    for m in memories:
        if m.get("category") == "response_style":
            always_include.append(m)
            continue
        score = len(q_words & set(m["text"].lower().split())) + m.get("confidence", 0.5)
        if score >= threshold:
            scored.append((m, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return "\n".join(f"- {m['text']}" for m in always_include + [m for m, _ in scored[:top_k]])
```

**Result**  
사용자별 context가 대화에 걸쳐 누적되어, 반복 설명 없이도 일관된 품질의 답변을 제공할 수 있게 되었습니다.

---

## 4. Performance Optimization

### 채팅 pipeline 비동기 처리

Context 로드 → 파일 처리 → RAG 검색 → 웹 검색 → LLM streaming의 단계적 pipeline에서, 각 단계의 blocking I/O를 `asyncio.to_thread`로 이벤트 루프에서 비블로킹 처리해 다른 요청을 차단하지 않도록 했습니다.

LLM streaming 응답은 동기 generator이므로 `await asyncio.to_thread(next, stream_response, None)` 패턴으로 한 chunk씩 이벤트 루프에 넘겨 SSE로 전달했습니다.

### Multi-LLM routing

GPT-nano, Keural, Qwen 등 여러 LLM backend를 지원합니다. 모델명을 인자로 받아 적절한 streaming 함수를 반환하는 factory pattern으로, 모델 추가 시 pipeline 코드를 수정하지 않아도 됩니다.

```python
selected_generate_stream_response = format_generate_stream_response(model=model)
stream_response = selected_generate_stream_response(model=model, prompt=prompt, system_prompt=system_prompt)
```

---

## 5. 그 외 기여

- **Citation 시스템:** RAG 검색 결과의 출처를 응답에 인용 마커로 연결하는 로직 구현 및 source ID 매핑 버그 수정
- **Web Search context 통합:** 웹 검색 결과를 SSE streaming 도중 실시간으로 prompt에 주입하는 로직 구현
- **Speech-to-Text(STT) API:** 음성을 텍스트로 변환하는 STT 로직 구현 및 테스트 완료 (미배포)
- **채팅 타이틀 자동 생성:** 대화 내용을 바탕으로 채팅방 제목을 자동 생성하는 로직 구현
- **System prompt 보호:** system prompt 내용이 LLM 응답에 노출되지 않도록 user/system prompt 분리 처리
