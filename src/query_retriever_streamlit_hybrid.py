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
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# 0)  상수 설정
# ---------------------------------------------------------------------------
FETCH_K = 100
TOP_K = 5
THRESHOLD = 0.6  # 임계값 상수

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

    # 원본 텍스트 (줄바꿈 포함)
    bm25_documents = df.apply(lambda row: f"제목: {row['제목']}\n\n본문: {row['본문_원본']}", axis=1).tolist()

    # BM25 토큰화를 위해서는 줄바꿈과 상관없이 단어 단위로 분리
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
    2) Dense 임베딩 검색 (FETCH_K개 문서를 검색)
    3) 두 스코어를 min-max 정규화한 후 선형 결합하여 최종 상위 문서를 반환
    """
    tokenized_query = query.split()

    # BM25 스코어 계산 및 정규화
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    bm25_scores_norm = bm25_scores

    # 임베딩 검색 (FETCH_K개 문서 검색)
    query_embedding = embedding_model.embed_query(query)
    D, I = dense_index.search(np.array([query_embedding]), FETCH_K)

    # 거리 -> 유사도로 변환 (유사도는 [0, 1] 범위에 가깝게)
    similarity_scores = 1 - (D / np.max(D))
    similarity_scores = similarity_scores.flatten()

    # BM25에서 dense 검색으로 반환된 문서 인덱스에 해당하는 BM25 스코어 선택
    selected_bm25_scores = bm25_scores_norm[I[0]]

    # BM25 스코어의 음수 제거 (최소값의 절대값을 더함)
    min_bm25_score = np.min(selected_bm25_scores)
    selected_bm25_scores = selected_bm25_scores + abs(min_bm25_score)

    # --- 두 스코어에 대해 min-max 정규화 진행 ---
    sb_min = np.min(selected_bm25_scores)
    sb_max = np.max(selected_bm25_scores)
    selected_bm25_scores = (selected_bm25_scores - sb_min) / (sb_max - sb_min + 1e-8)

    sim_min = np.min(similarity_scores)
    sim_max = np.max(similarity_scores)
    similarity_scores = (similarity_scores - sim_min) / (sim_max - sim_min + 1e-8)
    # --------------------------------------------

    # Hybrid 점수 계산 (두 스코어 모두 [0, 1] 범위)
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
    Hybrid Search 후 상위 TOP_K개의 문서를 context로 하여 GPT-4에 전송
    """
    hybrid_results = hybrid_search(question)
    
    # GPT에 전달할 컨텍스트 생성 (THRESHOLD 이상 문서만 포함)
    top_docs = [doc for doc in hybrid_results[:TOP_K] if doc[1] >= THRESHOLD]

    if not top_docs:
        # 🔹 문서를 찾지 못했을 때 GPT에게 기본 메시지를 전달하도록 설정
        context_text = "관련 문서를 찾지 못했습니다. 너가 아는 내용으로 대답해줘"
    else:
        # 🔹 정상적으로 문서를 찾았을 경우 문서 리스트 생성
        context_list = []
        for idx, (doc_text, score, bm25_score, faiss_score) in enumerate(top_docs, start=1):
            snippet = f"[문서 {idx} | Hybrid 점수={score:.3f} | BM25={bm25_score:.3f} | FAISS={faiss_score:.3f}]\n{doc_text}\n"
            context_list.append(snippet)
        context_text = "\n\n".join(context_list)
    
    # 🔹 Prompt 생성 및 GPT-4 호출
    prompt_input = {"question": question, "context": context_text}
    final_prompt = prompt.format(**prompt_input)
    result = llm.invoke(final_prompt)
    answer = result.content

    return answer, top_docs  # 필터링된 문서 리스트 반환


# ---------------------------------------------------------------------------
# 7) Streamlit UI + 검색 결과 엑셀 저장
# ---------------------------------------------------------------------------
st.title("📄 세무사 챗봇")
st.write("질문을 입력하면 AI가 문서를 참조하여 답변드립니다.")

with st.form("chat_form"):
    question = st.text_input("질문을 입력하세요:", placeholder="예: 배우자가 연구용역비를 받은 경우 배우자공제가 가능합니까?")
    submit_button = st.form_submit_button(label="질문하기")

if submit_button and question.strip():
    with st.spinner("답변 생성 중..."):
        answer, hybrid_results = generate_answer(question)

    st.subheader("💡 생성된 답변")
    st.write(answer)

    # TOP_K 문서 중 임계값 이상의 hybrid 점수를 가진 문서만 필터링
    filtered_docs = [doc for doc in hybrid_results[:TOP_K] if doc[1] >= THRESHOLD]
    if filtered_docs:
        st.subheader("🔍 참조한 문서")
        for idx, (doc_text, score, bm25_score, faiss_score) in enumerate(filtered_docs, start=1):
            with st.expander(f"문서 {idx} | Hybrid 점수: {score:.3f}"):
                st.write(doc_text[:2000])  # 문서가 길 경우 일부만 출력
                
    # # 검색된 모든 문서를 엑셀로 저장
    # import re
    # safe_question = re.sub(r'[\\/:*?"<>|]', '_', question)
    # desktop_path = os.path.join(os.path.expanduser("~"), "바탕화면")
    # excel_filename = f"{safe_question}.xlsx"
    # save_path = os.path.join(desktop_path, "RAG_엑셀", excel_filename)

    # df = pd.DataFrame({
    #     "질문": [question] * len(filtered_docs),
    #     "문서내용": [doc for (doc, _, _, _) in filtered_docs],
    #     "Hybrid 점수": [hybrid for (_, hybrid, _, _) in filtered_docs],
    #     "BM25 점수": [bm25 for (_, _, bm25, _) in filtered_docs],
    #     "FAISS 유사도 점수": [faiss for (_, _, _, faiss) in filtered_docs],
    # })

    # df.to_excel(save_path, index=False)
    # st.success(f"검색된 문서 전체를 엑셀로 저장했습니다: {save_path}")

