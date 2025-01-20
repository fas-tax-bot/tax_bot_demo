import os
import streamlit as st
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 프롬프트 파일 경로 설정
prompt_file_path = "src/prompt/prompt.txt"

# 텍스트 파일을 읽어와 프롬프트 텍스트로 저장
def load_prompt_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# 파일에서 프롬프트 텍스트 읽기
prompt_text = load_prompt_from_file(prompt_file_path)

# 임베딩 모델 생성 - text-embedding-3-small 사용
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터 저장소 로드 (allow_dangerous_deserialization 인자를 추가)
vectorstore = FAISS.load_local("vdb/faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# lambda_mult가 크면 정확도 향상, 작으면 다양성 향상
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 10, 'lambda_mult': 0.9})

# RAG 구성 요소 설정
prompt = PromptTemplate(
    input_variables=["question", "context"],  # 필요한 입력 변수 설정
    template=prompt_text  # 텍스트 파일에서 읽어온 프롬프트 텍스트
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit 앱 설정
st.set_page_config(page_title="RAG 기반 챗봇", page_icon="🤖", layout="wide")

st.title("📄 RAG 기반 세무사 챗봇")
st.write("질문을 입력하면 관련 문서를 검색하고 답변을 생성합니다.")

# 사용자 입력 폼
with st.form("chat_form"):
    question = st.text_input("질문을 입력하세요:", placeholder="예: 대학원생인 배우자가 2024년 6월에 연구용역비로 500만원을 받은 경우 배우자공제가 가능해?")
    submit_button = st.form_submit_button(label="질문하기")

if submit_button and question:
    # 문서 검색
    retrieved_documents = retriever.invoke(question)

    # 검색된 문서가 없을 경우 처리
    if not retrieved_documents:
        st.warning("관련 문서를 찾을 수 없습니다. 다른 질문을 입력해주세요.")
    else:
        # RAG를 사용하여 응답 생성
        with st.spinner("답변 생성 중..."):
            response = rag_chain.invoke(question)
            
        # 응답 출력
        st.subheader("💡 생성된 답변")
        st.write(response)
        
        # 리트리버된 문서를 Expand로 출력
        st.subheader("🔍 참조한 문서")
        for idx, doc in enumerate(retrieved_documents, 1):
            with st.expander(f"문서 {idx}: {doc.metadata.get('제목', '제목 없음')}"):
                st.write(f"**제목:** {doc.metadata.get('제목', '없음')}")
                st.write(f"**본문:** {doc.page_content}")
                st.write(f"**출처:** {doc.metadata.get('source', '출처 없음')}")
