import os
import streamlit as st
import numpy as np
import faiss
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# ✅ 유사도 임계치 (내적 점수가 0 이상인 문서만 사용)
FINAL_VIEWABLE_DOCUMENT_SCORE = 0.1

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 프롬프트 파일 로드
prompt_file_path = "src/prompt/prompt.txt"
def load_prompt_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
prompt_text = load_prompt_from_file(prompt_file_path)

# ✅ OpenAI 임베딩 모델 생성
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# ✅ 벡터 정규화 함수 추가 (L2 정규화하여 코사인 유사도를 내적으로 변환)
def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# ✅ FAISS 벡터 저장소 로드 (항상 코사인 유사도를 사용하도록 변경)
vectorstore = FAISS.load_local(
    "vdb/faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# ✅ FAISS 인덱스 강제 변환 (METRIC_INNER_PRODUCT 사용)
vectorstore.index.metric_type = faiss.METRIC_INNER_PRODUCT  # 무조건 내적 사용

# ✅ MMR 검색 적용
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 10, 'lambda_mult': 0.9}
)

# ✅ RAG 구성 요소 설정
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=prompt_text
)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ✅ Streamlit 앱 설정
st.set_page_config(page_title="세무사 챗봇", page_icon="🤖", layout="wide")

st.title("📄 세무사 챗봇")
st.write("질문을 입력하면 관련 문서를 검색하고 답변을 생성합니다.")

# ✅ 사용자 입력 폼
with st.form("chat_form"):
    question = st.text_input("질문을 입력하세요:", placeholder="예: 대학원생인 배우자가 2024년 6월에 연구용역비로 500만원을 받은 경우 배우자공제가 가능해?")
    submit_button = st.form_submit_button(label="질문하기")

if submit_button and question:
    # ✅ FAISS를 사용하여 문서 검색 (유사도 점수 포함)
    retrieved_documents_with_scores = vectorstore.similarity_search_with_score(
        question,
        search_type="mmr",
        search_kwargs={
            "k": 5,          # 최종 반환할 문서 수
            "fetch_k": 10,   # MMR 계산에 사용할 후보 수
            "lambda_mult": 0.9
        }
    )

    # 검색된 문서가 없을 경우 처리
    if not retrieved_documents_with_scores:
        st.warning("관련 문서를 찾을 수 없습니다. 다른 질문을 입력해주세요.")
    else:
        # ✅ RAG를 사용하여 응답 생성
        with st.spinner("답변 생성 중..."):
            response = rag_chain.invoke(question)

        # ✅ 답변 출력
        st.subheader("💡 생성된 답변")
        st.write(response)

        # ✅ 코사인 유사도 변환 (내적 그대로 사용)
        filtered_documents = [(doc, distance) for doc, distance in retrieved_documents_with_scores if distance >= FINAL_VIEWABLE_DOCUMENT_SCORE]

        # ✅ 참조한 문서 출력
        st.subheader("🔍 참조한 문서")
        if not filtered_documents:
            st.info(f"🔍 문서가 없습니다. (출처 없음)")
        else:
            for idx, (doc, similarity) in enumerate(filtered_documents, 1):
                with st.expander(f"문서 {idx}: {doc.metadata.get('제목', '제목 없음')} (유사도: {similarity:.3f})"):
                    st.write(f"**본문:** {doc.metadata.get('본문_원본', '없음')}")
                    st.write(f"**출처:** {doc.metadata.get('source', '출처 없음')}")
