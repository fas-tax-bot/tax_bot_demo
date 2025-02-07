import os
import json
import streamlit as st
from dotenv import load_dotenv
from GPT_Assistant import process_message  # GPT Assistant의 메시지 처리 함수 호출

# ---------------------------------------------------------------------------
# 1) Streamlit UI
# ---------------------------------------------------------------------------
st.title("📄 세무사 챗봇")
st.write("질문을 입력하면 AI가 답변을 제공합니다.")

# 질문 입력 폼
with st.form("chat_form"):
    question = st.text_input("질문을 입력하세요:", placeholder="예: 배우자가 연구용역비를 받은 경우 배우자공제가 가능합니까?")
    submit_button = st.form_submit_button(label="질문하기")

# 질문을 입력하고 버튼을 눌렀을 때
if submit_button and question.strip():
    with st.spinner("답변 생성 중..."):
        response_json = process_message(question)  # GPT Assistant 호출
        response_data = json.loads(response_json)  # JSON 변환

        # 🔹 AI 응답 출력
        st.subheader("💡 생성된 답변")
        for message in response_data["messages"]:
            if message["role"] == "assistant":
                st.write(message["message"])
