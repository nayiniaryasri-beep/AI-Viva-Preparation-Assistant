
import streamlit as st
import os
import time

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from google import genai


# =====================================================
# LOAD GEMINI API KEY
# =====================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY is not found in your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="AI Viva Preparation Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Viva Preparation Assistant")

st.write(
    "Upload your project report and practice viva questions "
    "using AI and RAG."
)


# =====================================================
# PDF UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "📄 Upload your Project Report",
    type=["pdf"]
)


if uploaded_file is not None:

    # =================================================
    # READ PDF
    # =================================================

    try:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:

        st.error(f"❌ Error reading PDF: {e}")
        st.stop()


    if not text.strip():

        st.error(
            "❌ No readable text was found in this PDF."
        )

        st.stop()


    st.success(
        "✅ Project report uploaded successfully!"
    )


    # =================================================
    # SPLIT TEXT INTO CHUNKS
    # =================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)


    # =================================================
    # CREATE VECTOR DATABASE
    # =================================================

    with st.spinner(
        "🔄 Processing project content..."
    ):

        try:

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vector_db = Chroma.from_texts(
                chunks,
                embedding=embeddings
            )

        except Exception as e:

            st.error(
                f"❌ Error creating vector database: {e}"
            )

            st.stop()


    st.success(
        "✅ Project content processed successfully!"
    )


    # =================================================
    # DIFFICULTY SELECTION
    # =================================================

    difficulty = st.selectbox(
        "🎯 Select Viva Difficulty",
        ["Easy", "Medium", "Hard"]
    )


    # =================================================
    # GENERATE VIVA QUESTION
    # =================================================

    if st.button(
        "❓ Generate Viva Question",
        use_container_width=True
    ):

        # Retrieve relevant project information

        docs = vector_db.similarity_search(
            "project introduction objectives "
            "problem statement methodology "
            "technologies modules implementation "
            "results conclusion",
            k=3
        )


        context = "\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )


        # =================================================
        # QUESTION PROMPT
        # =================================================

        prompt = f"""
You are an AI viva examiner.

Based ONLY on the project information below,
generate ONE {difficulty} viva question.

PROJECT INFORMATION:
{context}

Rules:
1. Ask exactly ONE question.
2. The question must be related to the project.
3. Do not give the answer.
4. Make the question suitable for a college student.
5. Do not create a question unrelated to the project.
"""


        question = None


        # =================================================
        # CALL GEMINI
        # =================================================

        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )


                if response and response.text:

                    question = response.text.strip()

                    break

                else:

                    st.error(
                        "❌ Gemini returned an empty response."
                    )

                    break


            except Exception as e:

                error_message = str(e)


                # -----------------------------------------
                # QUOTA ERROR
                # -----------------------------------------

                if (
                    "429" in error_message
                    or "RESOURCE_EXHAUSTED" in error_message
                ):

                    st.error(
                        "❌ Gemini API quota exceeded.\n\n"
                        "Please wait until your quota resets "
                        "and try again."
                    )

                    break


                # -----------------------------------------
                # SERVER BUSY
                # -----------------------------------------

                elif "503" in error_message:

                    if attempt < 2:

                        wait_time = 5 * (attempt + 1)

                        st.warning(
                            f"⚠️ Gemini is busy. "
                            f"Retrying in {wait_time} seconds..."
                        )

                        time.sleep(wait_time)

                    else:

                        st.error(
                            "⚠️ Gemini is temporarily busy. "
                            "Please try again after a few minutes."
                        )


                # -----------------------------------------
                # OTHER ERROR
                # -----------------------------------------

                else:

                    st.error(
                        f"❌ Gemini Error:\n\n{error_message}"
                    )

                    break


        # =================================================
        # DISPLAY QUESTION
        # =================================================

        if question:

            st.session_state["question"] = question

            st.subheader(
                "❓ Viva Question"
            )

            st.info(question)


# =====================================================
# ANSWER SECTION
# =====================================================

if "question" in st.session_state:

    st.divider()

    st.subheader(
        "✍️ Your Answer"
    )


    answer = st.text_area(
        "Enter your answer below:",
        height=150,
        placeholder="Type your viva answer here..."
    )


    # =================================================
    # EVALUATE ANSWER
    # =================================================

    if st.button(
        "📊 Evaluate Answer",
        use_container_width=True
    ):

        if not answer.strip():

            st.warning(
                "⚠️ Please enter your answer first."
            )

        else:

            # =================================================
            # EVALUATION PROMPT
            # =================================================

            evaluation_prompt = f"""
You are an AI viva examiner.

VIVA QUESTION:
{st.session_state["question"]}

STUDENT ANSWER:
{answer}

Evaluate the student's answer.

Give the result in exactly this format:

Score: X/10

What is correct:
- Mention the points answered correctly.

What is missing:
- Mention important points that are missing.

How to improve:
- Give simple suggestions to improve the answer.

Model Answer:
- Give a simple and correct answer to the viva question.

Keep the explanation suitable for a college student.
"""


            result = None


            # =================================================
            # CALL GEMINI FOR EVALUATION
            # =================================================

            for attempt in range(3):

                try:

                    response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=evaluation_prompt
                    )


                    if response and response.text:

                        result = response.text.strip()

                        break

                    else:

                        st.error(
                            "❌ Gemini returned an empty evaluation."
                        )

                        break


                except Exception as e:

                    error_message = str(e)


                    # -----------------------------------------
                    # QUOTA ERROR
                    # -----------------------------------------

                    if (
                        "429" in error_message
                        or "RESOURCE_EXHAUSTED" in error_message
                    ):

                        st.error(
                            "❌ Gemini API quota exceeded.\n\n"
                            "Please wait until your quota resets "
                            "and try again."
                        )

                        break


                    # -----------------------------------------
                    # SERVER BUSY
                    # -----------------------------------------

                    elif "503" in error_message:

                        if attempt < 2:

                            wait_time = 5 * (attempt + 1)

                            st.warning(
                                f"⚠️ Gemini is busy. "
                                f"Retrying in {wait_time} seconds..."
                            )

                            time.sleep(wait_time)

                        else:

                            st.error(
                                "⚠️ Gemini is temporarily busy. "
                                "Please try again later."
                            )


                    # -----------------------------------------
                    # OTHER ERROR
                    # -----------------------------------------

                    else:

                        st.error(
                            f"❌ Gemini Error:\n\n{error_message}"
                        )

                        break


            # =================================================
            # DISPLAY EVALUATION
            # =================================================

            if result:

                st.subheader(
                    "📊 AI Evaluation"
                )

                st.markdown(result)

