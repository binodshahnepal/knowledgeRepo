import google.generativeai as genai
from core.config import GEMINI_API_KEY, DEFAULT_LLM_MODEL


class LLMService:
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing.")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def answer_from_context(self, question: str, contexts: list[dict]) -> str:
        context_text = "\n\n".join(
            [f"Source: {c['source']} | Chunk: {c['chunk_id']}\n{c['text']}" for c in contexts]
        )

        prompt = f"""
You are a reliable enterprise knowledge assistant.

Use ONLY the provided context to answer the user's question.
Do not invent facts.
If the answer is not clearly available, say:
"I could not find a reliable answer in the uploaded sources."

Context:
{context_text}

Question:
{question}

Instructions:
- Be accurate
- Be concise but helpful
- Mention source names where useful
"""
        response = self.model.generate_content(prompt)
        return getattr(response, "text", "").strip() or "No answer returned."

    def summarize_text(self, text: str) -> str:
        prompt = f"""
Summarize the following content for a business user.
Focus on the main purpose, important themes, and useful insights.

Content:
{text}
"""
        response = self.model.generate_content(prompt)
        return getattr(response, "text", "").strip() or "No summary returned."

    def generate_sql(self, question: str, schema_text: str) -> str:
        prompt = f"""
You are a SQL assistant.

Generate ONLY one safe read-only SQL query.
Rules:
- Only SELECT
- No comments
- No markdown
- No explanation
- No INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, EXEC, MERGE

Schema:
{schema_text}

Question:
{question}
"""
        response = self.model.generate_content(prompt)
        sql = getattr(response, "text", "").strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql

    def explain_table_result(self, question: str, result_preview: str) -> str:
        prompt = f"""
The user asked:
{question}

Here is the query result preview:
{result_preview}

Explain the result clearly in plain language for a business user.
"""
        response = self.model.generate_content(prompt)
        return getattr(response, "text", "").strip() or "Could not explain result."