import os
import streamlit as st
import numpy as np
import faiss
import pandas as pd
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# LangChain / Custom modules
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# -----------------------------------------------------------------------------
# 1) 설정 상수
# -----------------------------------------------------------------------------
ALPHA = 0   # BM25와 FAISS 비중 (0 => FAISS 100%, 1 => BM25 100%)
RAG_TOP_K = 5      # 최종적으로 LLM(RAG)에 전달할 문서 개수
BM25_TOP_K = RAG_TOP_K // 2     # BM25 검색에서 상위 몇 개를 선택할지
FAISS_TOP_K = 10    # FAISS(MMR)에서 상위 몇 개를 선택할지
FINAL_VIEWABLE_DOCUMENT_SCORE = 0.5 # 보여지는 문서의 기준점수

# -----------------------------------------------------------------------------
# 2) .env 로드 & OpenAI API
# -----------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
os.environ["OPENAI_API_KEY"] = api_key

# -----------------------------------------------------------------------------
# 3) Prompt 파일 로드
# -----------------------------------------------------------------------------
prompt_file_path = "src/prompt/prompt.txt"
def load_prompt_from_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

prompt_text = load_prompt_from_file(prompt_file_path)

# -----------------------------------------------------------------------------
# 4) FAISS 로드 (METRIC_INNER_PRODUCT 강제)
# -----------------------------------------------------------------------------
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.load_local(
    "vdb/faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
vectorstore.index.metric_type = faiss.METRIC_INNER_PRODUCT

# -----------------------------------------------------------------------------
# 5) BM25 인덱스 (엑셀: "제목 + 본문_원본")
# -----------------------------------------------------------------------------
excel_file = "data_source/세무사 데이터전처리_20250116.xlsx"
df = pd.read_excel(excel_file)

bm25_documents = df.apply(lambda row: f"{row['제목']} {row['본문_원본']}", axis=1).tolist()
bm25_tokenized_docs = [doc.split() for doc in bm25_documents]
bm25 = BM25Okapi(bm25_tokenized_docs)

# -----------------------------------------------------------------------------
# 정규화를 위한 함수 (Min-Max)
# -----------------------------------------------------------------------------
def min_max_normalize(value, min_v, max_v):
    if max_v == min_v:
        return 0.0
    return (value - min_v) / (max_v - min_v)

# -----------------------------------------------------------------------------
# 6) 하이브리드 검색 함수
#    BM25_TOP_K + FAISS_TOP_K => 점수 합산 => 상위 RAG_TOP_K 문서 반환
#    여기서, BM25 점수를 0~1 범위로 정규화
# -----------------------------------------------------------------------------
def hybrid_search(query: str):
    """
    Returns a list of (doc_text, final_score) sorted by descending score.
    """
    print(f" - 입력: {query}")
    # 1) 전체 문서 BM25 점수 구하기
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)  # 모든 문서 BM25 점수
    doc_bm25_map = dict(zip(bm25_documents, bm25_scores))
    
    # 2) BM25 점수 Min-Max 정규화 (0~1)
    min_b = min(doc_bm25_map.values())
    max_b = max(doc_bm25_map.values()) if doc_bm25_map else 0.0
    
    normalized_bm25_map = {
        doc: min_max_normalize(score, min_b, max_b)
        for doc, score in doc_bm25_map.items()
    }

    # 3) 그중 상위 k => [(문서텍스트, 정규화된BM25점수), ...]
    bm25_top = sorted(
        normalized_bm25_map.items(),
        key=lambda x: x[1],
        reverse=True
    )[:BM25_TOP_K]

    # 4) FAISS (MMR) 검색 (with score)
    # faiss_results_with_scores = vectorstore.similarity_search_with_score(
    #     query,
    #     search_type="mmr",
    #     search_kwargs={
    #         "k": FAISS_TOP_K,
    #         "fetch_k": 10,
    #         "lambda_mult": 0.9
    #     }
    # )
    faiss_results_with_scores = vectorstore.similarity_search_with_score(
        query,
        k=FAISS_TOP_K
    )

    # 5) 하이브리드 점수 합산
    doc_score_map = {}

    # (A) 먼저 BM25 상위 k개 문서를 doc_score_map 에 반영
    for doc_text, bm25_val in bm25_top:
        doc_score_map[doc_text] = bm25_val
        print(f"[BM25] normalized={bm25_val:.3f} | doc_text={doc_text[:20]}...")

    # (B) FAISS 결과 (distance=낮을수록 유사) => similarity=1 - distance
    faiss_results_with_scores_sorted = sorted(
        faiss_results_with_scores, key=lambda x: 1.0 - x[1], reverse=True
    )
    
    for doc_obj, distance_value in faiss_results_with_scores_sorted:
        doc_text = doc_obj.page_content
        similarity = 1.0 - distance_value
        similarity = max(0.0, similarity)  # 음수 보정
        if similarity < 0.0:
            similarity = 0.0
        
        bm25_s = doc_score_map.get(doc_text, 0.0)  # 만약 BM25 상위 k에 없으면 0
        final_score = ALPHA * bm25_s + (1 - ALPHA) * similarity
        doc_score_map[doc_text] = final_score

        print(f"[FAISS] bm25_s={bm25_s:.3f} | sim={similarity:.3f} => final={final_score:.3f} | doc_text={doc_text[:20]}...")

    # 6) 상위 K 뽑아서 반환
    sorted_docs = sorted(doc_score_map.items(), key=lambda x: x[1], reverse=True)[:RAG_TOP_K]
    return sorted_docs

# -----------------------------------------------------------------------------
# 7) RAG(LLM QA) 구성
# -----------------------------------------------------------------------------
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=prompt_text
)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
parser = StrOutputParser()

def generate_answer(question: str):
    hybrid_results = hybrid_search(question)
    if not hybrid_results:
        return "관련 문서를 찾지 못했습니다.", []

    # context 생성
    context_list = []
    for idx, (doc_text, score) in enumerate(hybrid_results, 1):
        snippet = f"[문서 {idx} | 스코어={score:.3f}]\n{doc_text}\n"
        context_list.append(snippet)

    context_text = "\n\n".join(context_list)

    prompt_input = {"question": question, "context": context_text}
    result = llm.invoke(prompt.format(**prompt_input))
    answer = result.content

    return answer, hybrid_results

# -----------------------------------------------------------------------------
# 8) Streamlit 앱
# -----------------------------------------------------------------------------
st.set_page_config(page_title="세무사 챗봇 (하이브리드)", page_icon="🤖", layout="wide")
st.title("📄 세무사 챗봇 (BM25 + FAISS(MMR) 하이브리드, BM25=0~1 정규화)")

st.write("""
**하이브리드 검색 순서**  
1) **BM25** 전 문서 점수 -> **Min-Max 정규화(0~1)** -> 상위 k  
2) **FAISS(MMR)** top-k (with score)  
3) 두 점수 가중합(`ALPHA`)  
4) 최종 상위 k개를 LLM에 전달(RAG)  
""")

with st.form("chat_form"):
    question = st.text_input(
        "질문을 입력하세요:",
        placeholder="예: 대학원생인 배우자가 2024년 6월에 연구용역비 500만원을 받은 경우 배우자공제가 가능해?"
    )
    submit_button = st.form_submit_button(label="질문하기")

if submit_button and question.strip():
    with st.spinner("답변 생성 중..."):
        answer, top_docs = generate_answer(question)

    st.subheader("💡 생성된 답변")
    st.write(answer)

    st.subheader("🔍 참조한 문서")
    for idx, (doc_text, score) in enumerate(top_docs, start=1):
        if score >= FINAL_VIEWABLE_DOCUMENT_SCORE:  # ✅ 점수 기준 필터링
            if "본문:" in doc_text:
                doc_text = doc_text.replace("본문:", "\n\n본문:")  # ✅ "문서:" 앞에 줄바꿈 2개 추가

            with st.expander(f"문서 {idx} | 점수: {score:.3f}"):
                st.write(doc_text)  # 문서가 길 경우 일부만 출력

