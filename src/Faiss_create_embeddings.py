import os
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# 엑셀 파일 목록 및 시트 목록
excel_files = [
    ("data_source/세무사 데이터전처리_20250116.xlsx", ["Sheet1"]),
]

# 모든 데이터를 하나의 리스트에 저장
documents = []

# 파일별로 모든 시트를 읽어 Document로 변환
for file, sheets in excel_files:
    for sheet in sheets:
        # 시트 데이터 읽기
        df = pd.read_excel(file, engine='openpyxl', sheet_name=sheet)
        
        # NaN 데이터를 포함하는 행 개수 및 행 번호 출력
        df_nan = df[df["제목"].isna() | df["본문_원본"].isna()]
        print(f"🔍 NaN 데이터를 포함하는 행 개수: {len(df_nan)}")
        print(f"🔍 NaN이 포함된 행 번호: {df_nan.index.tolist()}")

        # 각 행을 Document로 변환하여 리스트에 추가
        for index, row in df.iterrows():
            # 제목과 본문_원본이 NaN이면 제외
            if pd.isna(row["제목"]) or pd.isna(row["본문_원본"]):
                continue  # NaN 데이터는 건너뜀
            
            # content는 제목과 본문_원본을 포함
            content = f"제목: {row['제목']}\n본문: {row['본문_원본']}"
            source = f"{file.replace('data_source/', '')}__{sheet}"

            # metadata에 모든 헤더 정보를 포함
            metadata = {
                "파일명": row["파일명"],
                "문서명": row["문서명"],
                "제목": row["제목"],
                "본문_원본": row["본문_원본"],
                "source": source
            }
            
            documents.append(Document(page_content=content, metadata=metadata))

# ✅ 엑셀에서 읽은 문서 개수 출력
print(f"✅ 엑셀에서 읽은 총 문서 개수: {len(documents)}")

# 임베딩 생성 - text-embedding-3-small 사용
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents=documents, embedding=embedding_model)

# 벡터 저장소와 메타데이터를 파일로 저장
vectorstore.save_local("vdb/faiss_index")
print("✅ 모든 시트의 임베딩이 성공적으로 저장되었습니다.")

# ✅ FAISS 벡터 개수 및 저장된 문서 개수 출력
print(f"✅ FAISS 벡터 개수 (dense_index): {vectorstore.index.ntotal}")
print(f"✅ FAISS 저장된 문서 개수 (docstore): {len(vectorstore.docstore._dict)}")
