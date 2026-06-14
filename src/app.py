"""
app.py — Gradio Interface
Vegetarian Food RAG System

Simple chat interface that wraps the full RAG pipeline.
Shows the answer, which sources were used, and their relevance scores.

Usage:
    pip install gradio
    python src/app.py
    Open http://localhost:7860 in your browser
"""

import gradio as gr
from generate import ask

# ─── Example questions pre-loaded in the UI ──────────────────────────────────
EXAMPLES = [
    "Does Tom Yum soup contain fish sauce or shrimp paste?",
    "What vegetarian options does Chipotle have?",
    "What hidden non-vegetarian ingredients should I watch for?",
    "Which Indian restaurants have vegetarian dishes?",
    "Are there vegetarian options at Moe's Southwest Grill?",
]


def run_query(question: str, source_filter: str, k: int) -> tuple[str, str]:
    """
    Called by Gradio on every submission.
    Returns (answer_text, sources_text) for the two output boxes.
    """
    if not question.strip():
        return "Please enter a question.", ""

    source_type = None
    if source_filter == "Menus only":
        source_type = "menu"
    elif source_filter == "Reddit only":
        source_type = "reddit"

    result = ask(question, k=int(k), source_type=source_type)

    # Clean bullet-point source list for the "Retrieved from" box
    # Uses the "sources" key added in generate.py for reliable attribution
    sources_text = "\n".join(f"• {s}" for s in result["sources"])

    return result["answer"], sources_text


# ─── Build Gradio UI ──────────────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Vegetarian Food Assistant") as demo:

        gr.Markdown("""
        # 🥗 Vegetarian Food Assistant
        Ask about vegetarian options at Tampa restaurants, hidden non-vegetarian
        ingredients, or safe dishes for dietary restrictions.
        *Answers are grounded in menu data and Reddit community discussions.*
        """)

        with gr.Row():
            with gr.Column(scale=3):
                question_box = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Does Tom Yum soup contain fish sauce?",
                    lines=2,
                )
            with gr.Column(scale=1):
                source_filter = gr.Dropdown(
                    label="Search in",
                    choices=["All sources", "Menus only", "Reddit only"],
                    value="All sources",
                )
                k_slider = gr.Slider(
                    label="Chunks to retrieve (k)",
                    minimum=3,
                    maximum=10,
                    step=1,
                    value=5,
                )

        submit_btn = gr.Button("Ask", variant="primary")

        with gr.Row():
            answer_box = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False,
            )
            sources_box = gr.Textbox(
                label="Sources used",
                lines=8,
                interactive=False,
            )

        gr.Examples(
            examples=EXAMPLES,
            inputs=question_box,
            label="Try these questions",
        )

        # Wire up the button and Enter key
        submit_btn.click(
            fn=run_query,
            inputs=[question_box, source_filter, k_slider],
            outputs=[answer_box, sources_box],
        )
        question_box.submit(
            fn=run_query,
            inputs=[question_box, source_filter, k_slider],
            outputs=[answer_box, sources_box],
        )

    return demo


if __name__ == "__main__":
    print("[app] Starting Vegetarian Food Assistant...")
    print("[app] Open http://localhost:7860 in your browser")
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())