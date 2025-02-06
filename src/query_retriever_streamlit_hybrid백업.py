import os
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings  # 커스텀 Embedding
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from sklearn.preprocessing import normalize, StandardScaler

# ---------------------------------------------------------------------------
# 0)  상수 설정
# ---------------------------------------------------------------------------
TOP_K = 577

# ---------------------------------------------------------------------------
# 1) .env 설정
# ---------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
os.environ["OPENAI_API_KEY"] = api_key

# ---------------------------------------------------------------------------
# 2) BM25 로딩
# ---------------------------------------------------------------------------
@st.cache_data
def load_bm25_index():
    excel_file = "data_source/세무사 데이터전처리_20250116.xlsx"
    df = pd.read_excel(excel_file)

    bm25_documents = df.apply(lambda row: f"{row['제목']} {row['본문_원본']}", axis=1).tolist()
    bm25_tokenized_docs = [doc.split() for doc in bm25_documents]
    bm25 = BM25Okapi(bm25_tokenized_docs)

    print(f"✅ BM25 문서 개수: {len(bm25_documents)}")
    return bm25, bm25_documents

bm25, bm25_documents = load_bm25_index()

# ---------------------------------------------------------------------------
# 3) FAISS (임베딩) 로딩
# ---------------------------------------------------------------------------
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_resource
def load_embedding_index():
    """
    LangChain Community FAISS를 사용하여 벡터 저장소 로드
    """
    try:
        vectorstore = FAISS.load_local(
            "vdb/faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True
        )
        print("✅ 기존 FAISS 임베딩 인덱스 로드 완료")
    except:
        print("❌ FAISS 인덱스 파일 없음, 새로 생성 필요")
        vectorstore = FAISS.from_documents([], embedding_model)
        vectorstore.save_local("vdb/faiss_index")

    return vectorstore

vectorstore = load_embedding_index()
dense_index = vectorstore.index

# ---------------------------------------------------------------------------
# 4) FAISS Sparse Index (BM25 점수 기반)
# ---------------------------------------------------------------------------
dimension = len(bm25_documents)
sparse_index = faiss.IndexFlatL2(dimension)

bm25_scores_matrix = np.array([bm25.get_scores(doc.split()) for doc in bm25_documents])
bm25_scores_matrix = normalize(bm25_scores_matrix, norm="l2", axis=1)
sparse_index.add(bm25_scores_matrix)

# ---------------------------------------------------------------------------
# 5) Hybrid Search (BM25 + Dense)
# ---------------------------------------------------------------------------


def hybrid_search(query: str, alpha=0.5):
    """
    1) BM25 전체 문서 유사도
    2) Dense 임베딩 검색 (TOP_K)
    3) 두 점수를 혼합하여 최종 상위 문서 반환
    """
    tokenized_query = query.split()

    # BM25 스코어
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    scaler = StandardScaler()
    bm25_scores_norm = scaler.fit_transform(bm25_scores.reshape(-1,1)).flatten()

    # 임베딩 검색
    query_embedding = embedding_model.embed_query(query)
    D, I = dense_index.search(np.array([query_embedding]), TOP_K)

    # 거리 -> 유사도
    similarity_scores = 1 - (D / np.max(D))
    similarity_scores = similarity_scores.flatten()

    # BM25에서 FAISS 검색된 문서만 선택
    selected_bm25_scores = bm25_scores_norm[I[0]]

    # Hybrid 점수 계산
    hybrid_score = alpha * selected_bm25_scores + (1 - alpha) * similarity_scores
    sorted_indices = np.argsort(-hybrid_score)
    final_results = [(bm25_documents[I[0][idx]], hybrid_score[idx]) for idx in sorted_indices]

    return final_results

# ---------------------------------------------------------------------------
# 6) RAG(LLM QA) 구성
# ---------------------------------------------------------------------------
prompt_file_path = "src/prompt/prompt.txt"
def load_prompt_from_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

prompt_text = load_prompt_from_file(prompt_file_path)

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=prompt_text
)

# ✅ GPT-4 모델 사용 (주의: 실제로 gpt-4 접근 가능해야 함)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
parser = StrOutputParser()

def generate_answer(question: str):
    """
    Hybrid Search -> 상위 문서 -> Context 생성 -> GPT-4에 Prompt
    """
    hybrid_results = hybrid_search(question)
    if not hybrid_results:
        return "관련 문서를 찾지 못했습니다.", []

    # (1) context 만들기
    context_list = []
    for idx, (doc_text, score) in enumerate(hybrid_results, start=1):
        snippet = f"[문서 {idx} | 스코어={score:.3f}]\n{doc_text}\n"
        context_list.append(snippet)
    context_text = "\n\n".join(context_list)

    # (2) Prompt 생성
    prompt_input = {"question": question, "context": context_text}
    final_prompt = prompt.format(**prompt_input)

    # (3) GPT-4 호출
    result = llm.invoke(final_prompt)
    answer = result.content

    return answer, hybrid_results

# ---------------------------------------------------------------------------
# 7) Streamlit UI
# ---------------------------------------------------------------------------
st.title("📄 세무사 챗봇 (Hybrid)")
st.write("질문을 입력하면 Hybrid Search 후 GPT-4 모델이 답변을 생성합니다.")

with st.form("chat_form"):
    question = st.text_input("질문을 입력하세요:", placeholder="예: 배우자가 연구용역비를 받은 경우 배우자공제가 가능합니까?")
    alpha = st.slider("Hybrid 가중치 (BM25 vs Dense)", 0.0, 1.0, 0.5, 0.1)
    submit_button = st.form_submit_button(label="질문하기")

if submit_button and question.strip():
    with st.spinner("답변 생성 중..."):
        answer, top_docs = generate_answer(question)

    st.subheader("💡 생성된 답변")
    st.write(answer)

    st.subheader("🔍 참조한 문서 (상위 10건)")
    for idx, (doc_text, score) in enumerate(top_docs, start=1):
        with st.expander(f"문서 {idx} | Hybrid 점수: {score:.3f}"):
            st.write(doc_text[:2000])
