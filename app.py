import os
import json
import re
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

# Get API key from local .env first
API_KEY = os.getenv("GEMINI_API_KEY")

# If running on Streamlit Cloud, get it from Streamlit Secrets
if not API_KEY:
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        API_KEY = None

# Get model name
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

try:
    MODEL_NAME = st.secrets.get("GEMINI_MODEL", MODEL_NAME)
except Exception:
    pass

# Stop only if the API key is missing everywhere
if not API_KEY:

    st.error(
        "GEMINI_API_KEY is missing. "
        "Please configure it in Streamlit Secrets."
    )

    st.stop()

client = genai.Client(api_key=API_KEY)

st.set_page_config(
    page_title="EcoScientist AI",
    page_icon="🌱",
    layout="wide"
)

# ============================================================
# KNOWLEDGE BASE
# ============================================================

@st.cache_data
def load_knowledge():

    try:
        with open("knowledge.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            st.error("knowledge.json must contain a list of knowledge items.")
            st.stop()

        return data

    except FileNotFoundError:
        st.error("knowledge.json was not found in the project folder.")
        st.stop()

    except json.JSONDecodeError:
        st.error("knowledge.json contains invalid JSON.")
        st.stop()


knowledge = load_knowledge()


def document_text(item):
    """
    Convert one knowledge-base item into searchable text.
    Uses safe defaults so a missing optional field does not crash the app.
    """

    title = item.get("title", "")
    topic = item.get("topic", "")
    content = item.get("content", "")
    variables = item.get("variables", [])
    recommendation = item.get("recommendation", "")
    source = item.get("source", "")

    if isinstance(variables, list):
        variables_text = ", ".join(str(v) for v in variables)
    else:
        variables_text = str(variables)

    return (
        f"Title: {title}\n"
        f"Topic: {topic}\n"
        f"Content: {content}\n"
        f"Variables: {variables_text}\n"
        f"Recommendation: {recommendation}\n"
        f"Source: {source}"
    )


# ============================================================
# SIMPLE LOCAL RETRIEVAL
# ============================================================
#
# IMPORTANT:
# We deliberately do NOT use Gemini embeddings here.
#
# This means opening the application does NOT consume
# embedding API calls.
#
# We perform lightweight keyword matching locally and then
# use Gemini only once to generate the final answer.
# ============================================================

STOP_WORDS = {
    "the", "and", "for", "with", "that", "this",
    "from", "are", "was", "were", "have", "has",
    "what", "when", "where", "which", "how",
    "why", "can", "could", "should", "would",
    "into", "about", "your", "you", "their",
    "there", "than", "then", "also", "very",
    "using", "use", "used", "does", "not",
    "but", "all", "any", "our", "its",
    "environment", "environmental"
}


def tokenize(text):
    """
    Convert text into useful lowercase keywords.
    """

    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return [
        word
        for word in words
        if len(word) > 2 and word not in STOP_WORDS
    ]


def calculate_relevance(query, item):
    """
    Simple local keyword-based relevance score.

    No API call is made here.
    """

    query_words = set(tokenize(query))

    if not query_words:
        return 0

    title = str(item.get("title", ""))
    topic = str(item.get("topic", ""))
    content = str(item.get("content", ""))
    recommendation = str(item.get("recommendation", ""))
    variables = item.get("variables", [])

    if isinstance(variables, list):
        variables_text = " ".join(str(v) for v in variables)
    else:
        variables_text = str(variables)

    # Give title/topic slightly higher importance.
    title_words = set(tokenize(title))
    topic_words = set(tokenize(topic))

    all_text = " ".join([
        title,
        topic,
        content,
        recommendation,
        variables_text
    ])

    document_words = set(tokenize(all_text))

    score = 0

    # Stronger weight for title matches.
    score += len(query_words.intersection(title_words)) * 5

    # Stronger weight for topic matches.
    score += len(query_words.intersection(topic_words)) * 4

    # Normal content matches.
    score += len(query_words.intersection(document_words))

    return score


def retrieve_documents(query, top_k=3):
    """
    Retrieve the most relevant knowledge-base items locally.
    """

    scored_items = []

    for item in knowledge:

        score = calculate_relevance(query, item)

        scored_items.append({
            "item": item,
            "score": score
        })

    scored_items.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Take top results.
    selected = scored_items[:top_k]

    results = []

    for result in selected:

        item = result["item"]

        results.append({
            "document": document_text(item),
            "score": result["score"],
            "source": item.get("source", "Unknown source"),
            "source_url": item.get("source_url", "")
        })

    return results


# ============================================================
# STRUCTURED JSON INPUT
# ============================================================

def parse_json_input(raw_input):

    try:

        data = json.loads(raw_input)

        if not isinstance(data, dict):
            return None, "JSON input must be an object."

        return data, None

    except json.JSONDecodeError as error:

        return None, f"Invalid JSON: {error}"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are EcoScientist AI, an environmental intelligence assistant.

Your purpose is to provide scientifically grounded and actionable
environmental recommendations using the provided knowledge base.

IMPORTANT RULES:

1. Use the retrieved knowledge as your primary evidence.

2. Do not invent studies, sources, citations, numerical estimates,
   experimental results, or scientific facts that are not supported
   by the available evidence.

3. Do not claim that a source supports information that it does not
   actually contain.

4. If the provided evidence is insufficient, clearly say so.

5. Analyze multiple environmental variables together whenever the
   available information supports this.

6. Consider relationships between:
   - soil
   - water
   - climate
   - biodiversity
   - land use
   - crops
   - temperature
   - rainfall
   - human activities

7. Clearly distinguish evidence-supported information from assumptions.

8. Do not provide generic recommendations without explaining the
   environmental reasoning behind them.

9. Avoid claiming certainty when local information is insufficient.

10. Mention when field measurements, local monitoring, or expert
    environmental/agricultural assessment would be appropriate.

11. Do not invent quantitative improvement percentages.

12. The retrieved knowledge may be incomplete. Use careful reasoning
    but stay within the evidence.

RESPONSE FORMAT:

## Environmental Assessment

Summarize the environmental situation.

## Variables Considered

Identify the important environmental variables and explain how
they interact.

## Recommendation

Give practical actions that could be considered.

## Scientific Reasoning

Explain why the recommendation may work and what conditions
could affect the result.

## Expected Metric Changes

List useful environmental metrics that should be monitored.

Do NOT invent numerical improvement values.

## Time Horizon

Discuss short-term, medium-term, and long-term considerations
when appropriate.

## Confidence

State High, Medium, or Low confidence.

Briefly explain why.

## Evidence

List only the retrieved sources that genuinely support the response.

If important information is missing, mention what additional
information would improve the assessment.
"""


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(user_query, retrieved_context, history):

    if retrieved_context:

        context_text = "\n\n".join(
            [
                f"""
RETRIEVED EVIDENCE {i + 1}

{item['document']}

Source:
{item['source']}

Source URL:
{item['source_url']}
"""
                for i, item in enumerate(retrieved_context)
            ]
        )

    else:

        context_text = "No relevant knowledge-base evidence was found."

    history_text = "\n".join(
        [
            f"{message['role']}: {message['content']}"
            for message in history[-6:]
        ]
    )

    return f"""
{SYSTEM_PROMPT}

CONVERSATION HISTORY:

{history_text}

KNOWLEDGE BASE EVIDENCE:

{context_text}

USER QUERY:

{user_query}

TASK:

Answer the user's environmental question using the evidence above.

If the knowledge base does not contain enough information,
be transparent about the limitation.

Do not fabricate sources or numerical results.
"""


# ============================================================
# GEMINI RESPONSE
# ============================================================

def generate_response(user_query, retrieved_context, history):

    prompt = build_prompt(
        user_query,
        retrieved_context,
        history
    )

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            if not response.text:
                return "The AI returned an empty response. Please try again."

            return response.text

        except Exception as error:

            error_message = str(error)

            # Temporary Gemini server overload
            if "503" in error_message or "UNAVAILABLE" in error_message:

                if attempt < max_attempts - 1:

                    wait_time = 3 * (attempt + 1)

                    time.sleep(wait_time)

                    continue

                return (
                    "🌱 Gemini is temporarily experiencing high demand. "
                    "Please try again in a few minutes."
                )

            # Other errors should still be shown
            raise


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🌱 EcoScientist AI")

st.markdown(
    """
### Evidence-Aware Environmental Intelligence

EcoScientist AI analyzes environmental problems using a curated
knowledge base and AI-assisted reasoning.
"""
)

st.info(
    "⚠️ This prototype provides AI-assisted environmental analysis. "
    "Recommendations should be verified using local data and, "
    "where appropriate, qualified environmental or agricultural "
    "professionals."
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Input Options")

    input_mode = st.radio(
        "Choose input format",
        ["Text", "JSON"]
    )

    st.divider()

    st.caption(
        f"Knowledge base entries: {len(knowledge)}"
    )

    st.caption(
        f"AI model: {MODEL_NAME}"
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# TEXT INPUT
# ============================================================

if input_mode == "Text":

    user_input = st.chat_input(
        "Describe your environmental problem..."
    )


# ============================================================
# JSON INPUT
# ============================================================

else:

    st.markdown("### 📊 Structured Environmental Input")

    default_json = {
        "soil_organic_carbon": 0.3,
        "rainfall": "low",
        "crop": "monoculture wheat",
        "region": "semi-arid",
        "temperature": "unknown",
        "land_use": "agricultural"
    }

    user_input = st.text_area(
        "Enter environmental information as JSON",
        value=json.dumps(
            default_json,
            indent=2
        ),
        height=250
    )

    submit_json = st.button(
        "🔬 Analyze JSON",
        use_container_width=True
    )

    if not submit_json:
        user_input = None


# ============================================================
# PROCESS USER INPUT
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Convert JSON input into a natural-language query
    # --------------------------------------------------------

    if input_mode == "JSON":

        parsed_data, error = parse_json_input(user_input)

        if error:

            st.error(error)
            st.stop()

        user_query = (
            "Analyze the following structured environmental data "
            "and provide an environmental assessment:\n\n"
            + json.dumps(
                parsed_data,
                indent=2
            )
        )

    else:

        user_query = user_input.strip()


    # --------------------------------------------------------
    # Prevent empty questions
    # --------------------------------------------------------

    if not user_query:

        st.warning("Please enter an environmental question.")
        st.stop()


    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })


    with st.chat_message("user"):

        st.markdown(user_query)


    # --------------------------------------------------------
    # Generate AI response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🌱 Analyzing environmental evidence..."
        ):

            try:

                # Local retrieval.
                # IMPORTANT:
                # This does NOT call Gemini.
                retrieved_context = retrieve_documents(
                    user_query,
                    top_k=3
                )


                # One Gemini generation call.
                answer = generate_response(
                    user_query,
                    retrieved_context,
                    st.session_state.messages
                )


                st.markdown(answer)


                # ------------------------------------------------
                # Show retrieved knowledge
                # ------------------------------------------------

                with st.expander(
                    "📚 Retrieved Knowledge"
                ):

                    if retrieved_context:

                        for i, item in enumerate(
                            retrieved_context
                        ):

                            st.markdown(
                                f"### Evidence {i + 1}"
                            )

                            st.markdown(
                                f"**Relevance score:** "
                                f"{item['score']}"
                            )

                            st.markdown(
                                item["document"]
                            )

                            if item["source_url"]:

                                st.markdown(
                                    f"[🔗 View Source]"
                                    f"({item['source_url']})"
                                )

                            st.divider()

                    else:

                        st.write(
                            "No relevant knowledge was found."
                        )


                # ------------------------------------------------
                # Save assistant response
                # ------------------------------------------------

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })


            except Exception as error:

                st.error(
                    "An error occurred while processing "
                    "your request."
                )

                st.exception(error)