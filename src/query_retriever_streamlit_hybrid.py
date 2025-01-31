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
# 1) 설정 상수들
# -----------------------------------------------------------------------------
# ALPHA: BM25와 FAISS 가중치 (0=FAISS 100%, 1=BM25 100%)
ALPHA = 0.5

# FAISS 스코어(Inner Product) 활용 시, 0~1 사이 값이 나올 수도 있고
# BM25 스코어는 0~10 또는 그 이상일 수도 있음 (문서 길이에 따라 상이)
# 실제로는 두 값을 정규화하는 편이 좋습니다(예: 0~1로)
# 여기서는 간단히 raw 점수에만 곱해서 합산

# -----------------------------------------------------------------------------
# 2) .env 파일 로드 & OpenAI API 설정
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
def load_prompt_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

prompt_text = load_prompt_from_file(prompt_file_path)

# -----------------------------------------------------------------------------
# 4) FAISS 벡터 저장소 로드
#    - Inner Product 모드 강제
# -----------------------------------------------------------------------------
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.load_local(
    "vdb/faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
# metric_type 설정 (내적)
vectorstore.index.metric_type = faiss.METRIC_INNER_PRODUCT

# -----------------------------------------------------------------------------
# 5) BM25 인덱스 구성 (엑셀의 "제목 + 본문_원본")
# -----------------------------------------------------------------------------
excel_file = "data_source/세무사 데이터전처리_20250116.xlsx"
df = pd.read_excel(excel_file)

# BM25 문서 리스트: 문자열 형태
bm25_documents = df.apply(lambda row: f"{row['제목']} {row['본문_원본']}", axis=1).tolist()
bm25_tokenized_docs = [doc.split() for doc in bm25_documents]
bm25 = BM25Okapi(bm25_tokenized_docs)

# -----------------------------------------------------------------------------
# 6) 하이브리드 검색 함수
#    - "BM25 top-k" & "FAISS top-k" => 점수를 합산
# -----------------------------------------------------------------------------
def hybrid_search(query: str, k=5):
    """하이브리드 검색을 수행해 (문자열, hybrid_score) 리스트를 반환."""
    tokenized_query = query.split()

    # 1) BM25 검색
    bm25_scores = bm25.get_scores(tokenized_query)
    # 정렬하여 상위 k
    # => [(문서텍스트, bm25_score), ...]
    bm25_top = sorted(
        zip(bm25_documents, bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    # 2) FAISS 검색
    # => [(Document, distance), ...]  distance: 내적값에 대한 (1 - similarity) 또는 유사
    faiss_results_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    # 실제 langchain_community.vectorstores.FAISS 는 distance가 "1 - cos_sim" 형태일 수도 있습니다.
    # 그러나 위에서 metric_type=METRIC_INNER_PRODUCT 를 강제했으므로
    # distance == 1 - inner_product or just the raw distance? 실제 구현 따라 다름
    # 일단 distance가 "낮을수록 유사"라고 가정 -> similarity = (1 - distance)
    # (만약 distance가 raw inner product라면, 아래 로직을 조정해야 합니다.)

    # 3) 하이브리드 점수 합산
    # - doc_score_map: { "문서텍스트": bm25_score (raw), ... }
    doc_score_map = {}

    # (A) 먼저 bm25_score를 저장(알파 곱하지 X)
    for doc_text, bm25_score_value in bm25_top:
        doc_score_map[doc_text] = bm25_score_value

    # (B) FAISS 결과와 합산
    #     Document.page_content를 문자열 키로 사용
    for doc_obj, distance_value in faiss_results_with_scores:
        # FAISS가 내적 유사도라고 가정하면,
        # similarity = (1 - distance) (만약 distance=1 - inner_product)
        # 또는 "distance" 자체가 inner_product 점수면, similarity=distance
        similarity = 1.0 - distance_value  # 일단 이런 식으로 가정

        doc_text = doc_obj.page_content
        bm25_s = doc_score_map.get(doc_text, 0.0)  # 없으면 0
        # 최종 = alpha * bm25_s + (1-alpha)*similarity
        # 여기서 alpha=BM25 비중, (1-alpha)=FAISS 비중
        final_score = ALPHA * bm25_s + (1.0 - ALPHA) * similarity
        doc_score_map[doc_text] = final_score

    # 4) 하이브리드 점수로 상위 k 추출
    # doc_score_map: { "문서텍스트": 최종점수, ... }
    sorted_docs = sorted(doc_score_map.items(), key=lambda x: x[1], reverse=True)[:k]

    return sorted_docs

# -----------------------------------------------------------------------------
# 7) RAG(LLM QA) 구성
#    - 하이브리드 검색 결과를 직접 LLM에 전달
# -----------------------------------------------------------------------------
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=prompt_text
)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
parser = StrOutputParser()

def generate_answer(question: str, top_k=5):
    """
    1) 하이브리드 검색 => 상위 k개 문서
    2) 문서들을 하나의 context 문자열로 합침
    3) LLM에 전달하여 답변 생성
    """
    hybrid_results = hybrid_search(question, k=top_k)
    if not hybrid_results:
        return "관련 문서를 찾지 못했습니다."

    # context 생성: 문서 내용 + 점수
    context_list = []
    for idx, (doc_text, score) in enumerate(hybrid_results, 1):
        snippet = f"[문서{idx} | 스코어={score:.3f}]\n{doc_text}\n"
        context_list.append(snippet)
    context_text = "\n\n".join(context_list)

    # RAG 프롬프트
    prompt_input = {
        "question": question,
        "context": context_text
    }
    # LLM 호출
    answer = llm(prompt.format(**prompt_input)).content
    return answer, hybrid_results

# -----------------------------------------------------------------------------
# 8) Streamlit 앱
# -----------------------------------------------------------------------------
st.set_page_config(page_title="세무사 챗봇", page_icon="🤖", layout="wide")

st.title("📄 세무사 챗봇 (BM25 + FAISS 하이브리드)")
st.write("BM25와 FAISS 점수를 가중합하여 최종 상위 문서를 LLM에 전달합니다.")

with st.form("chat_form"):
    question = st.text_input(
        "질문을 입력하세요:",
        placeholder="예: 대학원생인 배우자가 2024년 6월에 연구용역비로 500만원을 받은 경우 배우자공제가 가능해?"
    )
    submit_button = st.form_submit_button(label="질문하기")

if submit_button and question.strip():
    with st.spinner("답변 생성 중..."):
        answer, top_docs = generate_answer(question, top_k=5)

    st.subheader("💡 생성된 답변")
    st.write(answer)

    st.subheader("🔍 참조한 문서 (하이브리드 검색 상위 5개)")
    for idx, (doc_text, score) in enumerate(top_docs, start=1):
        with st.expander(f"문서 {idx} | 점수: {score:.3f}"):
            st.write(doc_text[:1000])  # 문서가 길 경우 일부만
