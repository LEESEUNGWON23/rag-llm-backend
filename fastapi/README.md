# be-fastapi — LLM 채팅 백엔드

[← 메인으로](../README.md)

채팅 API, LLM 라우팅, 대화 컨텍스트 관리를 담당하는 백엔드입니다.  
아래는 개발 과정에서 실제로 문제를 인식하고 개선한 사례들입니다.

---

## 1. 컨텍스트 압축: 단순 요약 프롬프트 → 구조화 + 대용량 병렬 처리

**문제 1 — 프롬프트 품질:** 초기 구현은 단순한 한 줄 지시문이었습니다.  
LLM이 무엇을 지켜야 하는지 기준이 없어 고유명사, 날짜, 수치 같은 핵심 정보가 압축 과정에서 유실되는 경우가 있었습니다.

**변경 전**
```python
def compress_context(combined_content: str) -> str:
    final_prompt = (
        "Please analyze and summarize the following information concisely. "
        "Extract only the key information that is essential and directly relevant. "
        f"Remove any redundant or unnecessary details:\n\n{combined_content}"
    )
    model = settings.GPTOSS_MODEL  # 모델 하드코딩
    return get_llm_content(model=model, prompt=final_prompt)
```

**변경 후 — 역할·보존 대상·출력 형식을 명시한 구조화 프롬프트**
```python
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
    target_len = len(combined_content) // 4  # 목표: 원본의 25%
    response = format_generate_response(model=model)(
        model=model,
        prompt=f"<history>\n{combined_content}\n</history>",
        system_prompt=CONTEXT_COMPRESSION_SYSTEM_PROMPT.format(target_len=target_len),
    )
    return get_llm_content(response).strip() or combined_content
```

---

**문제 2 — 대용량 컨텍스트:** 대화가 매우 길어지면 컨텍스트 전체가 LLM 최대 입력을 초과해 압축 자체가 실패했습니다.

**변경 후 — 청크 분할 + ThreadPoolExecutor 병렬 압축**
```python
def compress_context(combined_content: str, model: str = None) -> str:
    CHUNK_SIZE = 100_000
    if len(combined_content) > CHUNK_SIZE:
        chunks = [combined_content[i:i + CHUNK_SIZE]
                  for i in range(0, len(combined_content), CHUNK_SIZE)]
        with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as executor:
            compressed_chunks = list(executor.map(
                lambda c: compress_context(c, model=model), chunks
            ))
        return "\n".join(compressed_chunks)

    # 이하 단일 청크 처리...
```

---

## 2. 사용자 메모리: 없음 → 대화 기반 개인화 컨텍스트 추출

**문제:** 매 대화가 새로 시작되면 LLM은 사용자에 대한 정보가 전혀 없었습니다.  
반복적으로 같은 선호나 배경을 설명해야 하는 불편함이 있었고, 답변 품질도 일관되지 않았습니다.

**구현:** 10번째 턴마다 LLM이 대화에서 사용자 정보를 JSON으로 추출 → 저장 → 다음 대화 시 관련 메모리만 선별해 프롬프트에 주입합니다.

```python
# 대화에서 사용자 정보 추출 및 기존 메모리와 병합
def update_user_context(new_dialogue: str, existing_user_context: list[dict], model=None) -> list[dict]:
    response_text = get_llm_content(
        format_generate_response(model=model)(
            model=model,
            prompt=f"<Existing Profile>{json.dumps(existing_user_context)}</Existing Profile>"
                   f"<New Conversation>{new_dialogue}</New Conversation>",
            system_prompt=USER_CONTEXT_UPDATE_PROMPT,
        )
    )
    new_items = json.loads(response_text)

    # 중복 제거 후 병합, 최대 50개 유지
    merged_texts = {m["text"] for m in existing_user_context}
    for item in new_items:
        if item.get("text") not in merged_texts:
            existing_user_context.append(item)
            merged_texts.add(item["text"])

    return existing_user_context[-50:]


# 쿼리와 관련성 높은 메모리만 선별해 주입
def select_relevant_memories(query: str, memories, top_k: int = 5) -> str:
    q_words = set(query.lower().split())
    scored = []
    always_include = []  # response_style 카테고리는 항상 포함

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

---

## 3. 스트리밍 저장 실패: 클라이언트 끊김 시 대화 유실 → asyncio.shield로 보호

**문제:** 스트리밍 도중 클라이언트가 연결을 끊으면 `asyncio.CancelledError`가 발생합니다.  
`finally` 블록에서 대화를 저장하는 작업이 있었지만, CancelledError가 태스크를 취소시켜 저장이 완료되지 않는 경우가 생겼습니다.

**변경 전**
```python
finally:
    await _finalize_response(...)  # CancelledError 발생 시 중단될 수 있음
```

**변경 후**
```python
finally:
    # asyncio.shield로 감싸면 외부 CancelledError로부터 태스크를 보호
    finalize_task = asyncio.create_task(_finalize_response(...))
    await asyncio.shield(finalize_task)  # 취소돼도 finalize_task는 끝까지 실행

    if cancelled:
        raise asyncio.CancelledError
```

`asyncio.shield`는 `await` 자체는 취소될 수 있지만, 감싸진 태스크(`finalize_task`)는 취소되지 않습니다.  
저장 작업이 완료된 뒤에 `CancelledError`를 다시 raise해 정상적인 취소 흐름도 유지합니다.

---

## 4. 토큰 초과: 응답 오류 → 실시간 긴급 압축

**문제:** 대화가 길어지면 조립된 프롬프트가 모델의 최대 입력 토큰을 초과해 응답이 실패했습니다.

**변경 후** — 프롬프트 조립 후 토큰 수를 체크하고 75% 초과 시 즉시 컨텍스트를 압축 후 재조립합니다.

```python
prompt_token_count = await asyncio.to_thread(get_tokenize, model=model, prompt=prompt)

if prompt_token_count > model_max_input_token * 0.75:
    # 긴급 압축: 대화 이력만 압축하고 프롬프트 재조립
    conversation_context = await asyncio.to_thread(
        compress_context, conversation_context, model=model
    )
    prompt = DEFAULT_LLM_PROMPT.format(
        ..., conversation_context=conversation_context, ...
    )
```

75%를 기준으로 삼은 이유는 RAG 컨텍스트와 웹 검색 결과가 추가되는 여유분을 남겨두기 위해서입니다.
