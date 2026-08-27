from langchain_ollama import ChatOllama

from searching_strategy.hybrid_search import hybrid_search


MODEL = "llama3.2"

llm = ChatOllama(model=MODEL)


# ==================================================
# REWRITE FOLLOW-UP AS A STANDALONE QUESTION
# ==================================================

def rewrite_standalone_question(history, question):

    if not history:
        return question

    history_text = "\n".join(
        f"{turn['role']}: {turn['content']}"
        for turn in history
    )

    prompt = f"""
Given the conversation history and a follow-up question,
rewrite the follow-up as a standalone question that makes
sense without the history.

Reply with ONLY the rewritten question, nothing else.

Conversation history:
{history_text}

Follow-up question:
{question}

Standalone question:
"""

    response = llm.invoke([
        {
            "role": "user",
            "content": prompt,
        }
    ])

    return response.content.strip()


# ==================================================
# BUILD CODE CONTEXT
# ==================================================

def build_context(results):

    return "\n\n".join(

        f"""
File:
{result.metadata['file']}

Name:
{result.metadata['name']}

Type:
{result.metadata['type']}

Lines:
{result.metadata.get('start_line', 0)}
-
{result.metadata.get('end_line', 0)}

Code:
{result.page_content}
"""

        for result in results
    )


# ==================================================
# SHOW RETRIEVAL
# ==================================================

def print_retrieved_context(results):

    print(
        "\n========== RETRIEVED CONTEXT ==========\n"
    )

    for result in results:

        print(
            "----------------------------------------"
        )

        print(
            f"File: {result.metadata['file']}"
        )

        print(
            f"Name: {result.metadata['name']}"
        )

        print(
            f"Type: {result.metadata['type']}"
        )

        print("\nCode:")

        print(
            result.page_content
        )


# ==================================================
# ANSWER ONE QUESTION
# ==================================================

def answer_question(history, question, top_k=3):

    standalone_question = rewrite_standalone_question(
        history,
        question,
    )

    results = hybrid_search(
        standalone_question,
        top_k=top_k,
    )

    # print_retrieved_context(results)

    context = build_context(results)

    prompt = f"""
You are a codebase assistant.

Answer the user's question using ONLY
the provided code context.

Rules:

1. Give a direct answer.
2. If the answer is present in the context,
   do not say you don't know.
3. Cite the source file for every fact you use -
   if you draw information from multiple files,
   list all of them, not just one.
4. Include the function/class name when available.
5. Include line numbers when available.
6. Do not invent information.

Question:

{question}


Code context:

{context}


Answer:
"""

    response = llm.invoke(
        history + [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    return response.content


# ==================================================
# CHAT LOOP
# ==================================================

history = []

print(
    "Codebase RAG - ask a question "
    "(type 'exit' to quit)\n"
)

while True:

    question = input("Question: ").strip()

    if question.lower() in ("exit", "quit"):
        break

    if not question:
        continue

    answer = answer_question(history, question)

    print(
        "\n========== ANSWER ==========\n"
    )

    print(answer)

    print()

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
