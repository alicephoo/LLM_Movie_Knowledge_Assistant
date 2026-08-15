"""
Movie Knowledge Assistant

Streamlit Interface
"""

import streamlit as st

from src.rag import answer_movie_question

from src.monitoring import (
    initialize_monitoring,
    save_user_feedback
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Knowledge Assistant",
    page_icon="🎬",
    layout="centered"
)

initialize_monitoring()


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "answer" not in st.session_state:
    st.session_state.answer = None

if "question" not in st.session_state:
    st.session_state.question = None

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎬 Movie Knowledge Assistant")

st.write(
    """
Ask anything about movies.

Examples:
- Which movie should I watch if I liked Interstellar?
- Recommend Christopher Nolan movies.
- Find science fiction movies about space.
"""
)


# --------------------------------------------------
# User Input
# --------------------------------------------------

question = st.text_input(
    "Enter your movie question:"
)


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------

if st.button("Ask"):

    if question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching movies and generating answer..."
        ):

            answer = answer_movie_question(question)

        # Save answer in session state
        st.session_state.question = question
        st.session_state.answer = answer
        st.session_state.feedback_submitted = False


# --------------------------------------------------
# Display Answer
# --------------------------------------------------

if st.session_state.answer:

    st.subheader("Recommendation")

    st.write(
        st.session_state.answer
    )


    # --------------------------------------------------
    # User Feedback
    # --------------------------------------------------

    st.divider()

    st.write(
        "Was this answer helpful?"
    )

    feedback = st.radio(
        "Feedback",
        [
            "👍 Yes",
            "👎 No"
        ],
        horizontal=True,
        key="feedback"
    )


    if st.button("Submit Feedback"):

        feedback_value = (
            "positive"
            if feedback == "👍 Yes"
            else "negative"
        )

        save_user_feedback(
            question=st.session_state.question,
            answer=st.session_state.answer,
            feedback=feedback_value
        )

        st.session_state.feedback_submitted = True


    if st.session_state.feedback_submitted:

        st.success(
            "Thank you for your feedback! 🙏"
        )