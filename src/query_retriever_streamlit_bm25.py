import os
import streamlit as st
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# 1) .env 로드 & 환경 변수 설정
# ----------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
os.environ["OPENAI_API_KEY"] = api_key
TOP_K = 10

# ----------------------------------------------------------------------
# 2) BM25 인덱스 (엑셀 파일에서 "제목 + 본문_원본" 가져오기) → **초기화 시점에서 한 번만 수행**
# ----------------------------------------------------------------------
@st.cache_data  # ✅ Streamlit에서 캐싱하여 불필요한 재연산 방지
def load_bm25_index():
    excel_file = "data_source/세무사 데이터전처리_20250116.xlsx"
    df = pd.read_excel(excel_file)

    # ✅ 제목과 본문을 결합하여 검색 대상 문서 리스트 생성
    bm25_documents = df.apply(lambda row: f"{row['제목']} {row['본문_원본']}", axis=1).tolist()

    # ✅ BM25 토큰화 및 인덱스 생성
    bm25_tokenized_docs = [doc.split() for doc in bm25_documents]
    bm25 = BM25Okapi(bm25_tokenized_docs)

    # ✅ 검색된 문서 개수 출력
    print(f"✅ BM25 문서 개수: {len(bm25_documents)}")

    return bm25, bm25_documents

# ✅ 최초 실행 시 한 번만 실행, 이후에는 캐싱된 값 사용
bm25, bm25_documents = load_bm25_index()

# ----------------------------------------------------------------------
# 3) BM25 검색 함수 (최상위 k개 문서 반환)
# ----------------------------------------------------------------------
def bm25_search(query: str):
    """
    Returns a list of (doc_text, bm25_score) sorted by BM25 score.
    """
    tokenized_query = query.split()

    print(f"\n🔍 [BM25] 검색어: {query}")  # ✅ 입력된 검색어 출력

    # ✅ BM25 스코어 계산
    bm25_scores = bm25.get_scores(tokenized_query)

    # ✅ 모든 문서와 BM25 점수를 매핑
    doc_bm25_map = dict(zip(bm25_documents, bm25_scores))

    # ✅ 상위 k개 문서 선택
    top_docs = sorted(
        doc_bm25_map.items(),
        key=lambda x: x[1],
        reverse=True
    )[:TOP_K]

    # ✅ BM25 검색된 문서 출력 (디버깅용)
    print("\n⭐ [BM25] Top-k 문서:")
    for doc, score in top_docs:
        print(f"  - 문서: {doc[:50]}... | BM25 점수: {score:.3f}")

    return top_docs

# ----------------------------------------------------------------------
# 4) Streamlit UI 설정
# ----------------------------------------------------------------------
st.set_page_config(page_title="세무사 챗봇 (BM25)", page_icon="🤖", layout="wide")
st.title("📄 세무사 챗봇 (BM25 검색)")

st.write("질문을 입력하면 BM25를 사용하여 관련 문서를 검색하고 답변을 제공합니다.")

# ✅ 사용자 입력 폼
with st.form("chat_form"):
    question = st.text_input(
        "질문을 입력하세요:",
        placeholder="예: 대학원생인 배우자가 2024년 6월에 연구용역비 500만원을 받은 경우 배우자공제가 가능해?"
    )
    submit_button = st.form_submit_button(label="질문하기")

if submit_button and question.strip():
    # ✅ BM25 검색 실행
    retrieved_documents_with_scores = bm25_search(question)

    # 검색된 문서가 없을 경우 처리
    if not retrieved_documents_with_scores:
        st.warning("관련 문서를 찾을 수 없습니다. 다른 질문을 입력해주세요.")
    else:
        # ✅ 참조한 문서 출력
        st.subheader("🔍 참조한 문서")
        for idx, (doc_text, score) in enumerate(retrieved_documents_with_scores, start=1):
            with st.expander(f"문서 {idx} | BM25 점수: {score:.3f}"):
                st.write(doc_text[:2000])  # 문서가 길 경우 일부분만 출력
