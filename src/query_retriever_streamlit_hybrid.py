import os
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from sklearn.preprocessing import normalize  # StandardScaler 제거

# ---------------------------------------------------------------------------
# 0)  상수 설정
# ---------------------------------------------------------------------------
TOP_K = 574  # 검색 결과를 전체 문서 개수만큼 (577)

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
    1) BM25 전체 문서 유사도 계산
    2) Dense 임베딩 검색 (TOP_K)
    3) 두 스코어를 min-max 정규화한 후 선형 결합하여 최종 상위 문서를 반환
    """
    tokenized_query = query.split()

    # BM25 스코어 계산 (원시 점수를 그대로 사용)
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    # StandardScaler 제거: 원시 점수를 그대로 사용함
    bm25_scores_norm = bm25_scores

    # 임베딩 검색
    query_embedding = embedding_model.embed_query(query)
    D, I = dense_index.search(np.array([query_embedding]), TOP_K)

    # 거리 -> 유사도로 변환 (유사도는 [0, 1] 범위에 가까움)
    similarity_scores = 1 - (D / np.max(D))
    similarity_scores = similarity_scores.flatten()

    # BM25에서 FAISS 검색된 문서만 선택
    selected_bm25_scores = bm25_scores_norm[I[0]]

    # BM25 스코어의 음수를 없애기 위해 최소값의 절대값을 더함
    min_bm25_score = np.min(selected_bm25_scores)
    selected_bm25_scores = selected_bm25_scores + abs(min_bm25_score)

    # --- 두 스코어에 대해 min-max 정규화 진행 ---
    # BM25 스코어 정규화
    sb_min = np.min(selected_bm25_scores)
    sb_max = np.max(selected_bm25_scores)
    selected_bm25_scores = (selected_bm25_scores - sb_min) / (sb_max - sb_min + 1e-8)

    # similarity_scores 정규화
    sim_min = np.min(similarity_scores)
    sim_max = np.max(similarity_scores)
    similarity_scores = (similarity_scores - sim_min) / (sim_max - sim_min + 1e-8)
    # --------------------------------------------

    # Hybrid 점수 계산 (두 스코어 모두 [0, 1] 범위를 가지게 됨)
    hybrid_score = alpha * selected_bm25_scores + (1 - alpha) * similarity_scores
    sorted_indices = np.argsort(-hybrid_score)
    
    final_results = [
        (bm25_documents[I[0][idx]], hybrid_score[idx], selected_bm25_scores[idx], similarity_scores[idx])
        for idx in sorted_indices
    ]
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

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
parser = StrOutputParser()

def generate_answer(question: str):
    """
    Hybrid Search 후 상위 10개의 문서를 context로 하여 GPT-4에 전송
    """
    hybrid_results = hybrid_search(question)
    if not hybrid_results:
        return "관련 문서를 찾지 못했습니다.", []

    # 상위 10개 문서로 context 생성
    top_10_docs = hybrid_results[:10]
    context_list = []
    for idx, (doc_text, score, bm25_score, faiss_score) in enumerate(top_10_docs, start=1):
        snippet = f"[문서 {idx} | Hybrid 점수={score:.3f} | BM25={bm25_score:.3f} | FAISS={faiss_score:.3f}]\n{doc_text}\n"
        context_list.append(snippet)
    context_text = "\n\n".join(context_list)
    
    # Prompt 생성 및 GPT-4 호출
    prompt_input = {"question": question, "context": context_text}
    final_prompt = prompt.format(**prompt_input)
    result = llm.invoke(final_prompt)
    answer = result.content

    return answer, hybrid_results  # hybrid_results 전체 반환

# ---------------------------------------------------------------------------
# 7) Streamlit UI + 검색 결과 엑셀 저장
# ---------------------------------------------------------------------------
st.title("📄 세무사 챗봇 (Hybrid)")
st.write("질문을 입력하면 Hybrid Search 후 GPT-4 모델이 답변을 생성합니다. "
         "또한 검색된 문서를 엑셀로 저장합니다.")

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
    for idx, (doc_text, score, bm25_score, faiss_score) in enumerate(top_docs[:10], start=1):
        with st.expander(f"문서 {idx} | Hybrid 점수: {score:.3f}"):
            st.write(doc_text[:2000])  # 문서가 길 경우 일부만 출력

    # # 검색된 모든 문서를 엑셀로 저장
    # import re
    # safe_question = re.sub(r'[\\/:*?"<>|]', '_', question)
    # desktop_path = os.path.join(os.path.expanduser("~"), "바탕화면")
    # excel_filename = f"{safe_question}.xlsx"
    # save_path = os.path.join(desktop_path, "RAG_엑셀", excel_filename)

    # df = pd.DataFrame({
    #     "질문": [question] * len(top_docs),
    #     "문서내용": [doc for (doc, _, _, _) in top_docs],
    #     "Hybrid 점수": [hybrid for (_, hybrid, _, _) in top_docs],
    #     "BM25 점수": [bm25 for (_, _, bm25, _) in top_docs],
    #     "FAISS 유사도 점수": [faiss for (_, _, _, faiss) in top_docs],
    # })

    # df.to_excel(save_path, index=False)
    # st.success(f"검색된 문서 전체를 엑셀로 저장했습니다: {save_path}")
