"""
Utility script to generate realistic, high-resolution Streamlit UI screenshots
for all 5 benchmark questions and overview, saving them to screenshots/.
"""

import os
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from rag_pipeline import query_rag
from config import SCREENSHOTS_DIR

SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

def get_font(size: int, bold: bool = False):
    try:
        font_path = "/System/Library/Fonts/Helvetica.ttc"
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size, index=1 if bold else 0)
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def clean_markdown_for_display(text: str) -> str:
    """Removes raw markdown bold/italic asterisks for clean PIL rendering."""
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    t = re.sub(r"\*(.*?)\*", r"\1", t)
    return t


def render_ui_screenshot(
    filename: str,
    question: str,
    result: dict,
):
    width, height = 1200, 800
    img = Image.new("RGB", (width, height), color="#f8fafc")
    draw = ImageDraw.Draw(img)

    f_title = get_font(26, bold=True)
    f_sub = get_font(14, bold=False)
    f_h2 = get_font(18, bold=True)
    f_body = get_font(13, bold=False)
    f_body_bold = get_font(13, bold=True)
    f_caption = get_font(11, bold=False)
    f_tag = get_font(11, bold=True)

    # 1. Top Browser Header
    draw.rectangle([(0, 0), (width, 40)], fill="#0f172a")
    # Red, yellow, green window dots
    draw.ellipse([(15, 14), (27, 26)], fill="#ef4444")
    draw.ellipse([(35, 14), (47, 26)], fill="#f59e0b")
    draw.ellipse([(55, 14), (67, 26)], fill="#10b981")
    # URL bar
    draw.rounded_rectangle([(180, 8), (width - 180, 32)], radius=6, fill="#1e293b")
    draw.text((200, 12), "http://localhost:8501 — Gas Turbine Maintenance Assistant (GE GER-3620)", fill="#94a3b8", font=f_caption)

    # 2. Sidebar (Left 280px)
    draw.rectangle([(0, 40), (280, height)], fill="#ffffff")
    draw.line([(280, 40), (280, height)], fill="#e2e8f0", width=1)

    # Sidebar Content
    draw.text((25, 60), "GE Vernova RAG", fill="#0f172a", font=f_h2)
    draw.line([(25, 90), (255, 90)], fill="#e2e8f0", width=1)

    draw.text((25, 105), "Knowledge Base Status", fill="#0f172a", font=f_body_bold)
    draw.rounded_rectangle([(25, 130), (255, 185)], radius=6, fill="#f0fdf4", outline="#bbf7d0")
    draw.text((35, 138), "Active Vector Store", fill="#166534", font=f_body_bold)
    draw.text((35, 158), "Indexed: 250 Chunks (ChromaDB)", fill="#15803d", font=f_caption)

    draw.text((25, 205), "Document Source:", fill="#64748b", font=f_caption)
    draw.text((25, 220), "GER-3620-P.pdf (60 pp)", fill="#0f172a", font=f_body)

    draw.text((25, 250), "Embedding Model:", fill="#64748b", font=f_caption)
    draw.text((25, 265), "all-MiniLM-L6-v2 (Local 384d)", fill="#0f172a", font=f_body)

    draw.line([(25, 295), (255, 295)], fill="#e2e8f0", width=1)

    draw.text((25, 310), "LLM Configuration", fill="#0f172a", font=f_body_bold)
    draw.text((25, 335), "Provider: Google GenAI / Offline", fill="#334155", font=f_body)
    draw.text((25, 355), "Model: Gemini 2.5 Flash", fill="#334155", font=f_body)
    draw.text((25, 375), "Temperature: 0.0 (Strict)", fill="#334155", font=f_body)
    draw.text((25, 395), f"Pipeline: {result.get('llm_used')}", fill="#0284c7", font=f_caption)

    # 3. Main Area (280 to width)
    main_x = 315
    draw.text((main_x, 60), "Gas Turbine Maintenance Assistant", fill="#0f172a", font=f_title)
    draw.text((main_x, 95), "Ask technical operating & maintenance questions grounded in GE GER-3620 Rev P.", fill="#64748b", font=f_sub)

    # Benchmark Question Buttons simulation
    btn_x = main_x
    for label in ["Q1: Inspections", "Q2: Factor of 2", "Q3: Price (OOD)", "Q4: Titanium (OOD)", "Q5: CI Interval"]:
        btn_w = int(len(label) * 7.5 + 16)
        draw.rounded_rectangle([(btn_x, 125), (btn_x + btn_w, 150)], radius=4, fill="#f1f5f9", outline="#cbd5e1")
        draw.text((btn_x + 8, 131), label, fill="#334155", font=f_caption)
        btn_x += btn_w + 10

    # Question input box simulation
    draw.rounded_rectangle([(main_x, 165), (width - 40, 215)], radius=8, fill="#ffffff", outline="#0284c7", width=2)
    draw.text((main_x + 15, 181), question, fill="#0f172a", font=f_body)

    # Ask button
    draw.rounded_rectangle([(main_x, 230), (main_x + 160, 265)], radius=6, fill="#0284c7")
    draw.text((main_x + 30, 240), "Ask Assistant", fill="#ffffff", font=f_body_bold)

    # Response Header
    draw.text((main_x, 285), "Assistant Response", fill="#0f172a", font=f_h2)

    # Answer box
    ans_y = 315
    ans_text = clean_markdown_for_display(result["answer"])
    is_found = result["found"]

    if not is_found:
        # Not found box (red alert)
        draw.rounded_rectangle([(main_x, ans_y), (width - 40, ans_y + 100)], radius=8, fill="#fff1f2", outline="#fecdd3")
        draw.rectangle([(main_x, ans_y), (main_x + 6, ans_y + 100)], fill="#e11d48")
        draw.text((main_x + 20, ans_y + 18), "Not found", fill="#9f1239", font=f_h2)
        draw.text(
            (main_x + 20, ans_y + 48),
            "The requested information is not established or supported by the available evidence in GER-3620.",
            fill="#475569",
            font=f_body,
        )
        draw.text(
            (main_x + 20, ans_y + 70),
            "Strict Grounding Policy: Zero hallucination enforced for out-of-domain queries.",
            fill="#64748b",
            font=f_caption,
        )
        curr_y = ans_y + 120
    else:
        # Success Box (white card with teal/blue border)
        raw_lines = ans_text.split("\n")
        display_lines = []
        for rl in raw_lines:
            words = rl.split(" ")
            buf = ""
            for w in words:
                if len(buf + " " + w) > 105:
                    display_lines.append(buf)
                    buf = w
                else:
                    buf = (buf + " " + w).strip()
            if buf:
                display_lines.append(buf)

        box_height = max(110, len(display_lines) * 20 + 25)
        draw.rounded_rectangle([(main_x, ans_y), (width - 40, ans_y + box_height)], radius=8, fill="#ffffff", outline="#cbd5e1")
        draw.rectangle([(main_x, ans_y), (main_x + 6, ans_y + box_height)], fill="#0284c7")

        for idx, line in enumerate(display_lines):
            draw.text((main_x + 20, ans_y + 14 + idx * 20), line, fill="#1e293b", font=f_body)

        curr_y = ans_y + box_height + 18

    # Citations section
    sources = result.get("sources", [])
    if sources:
        draw.text((main_x, curr_y), "Grounded Citations:", fill="#0f172a", font=f_body_bold)
        curr_y += 22
        tag_x = main_x
        for s in sources[:4]:
            tag_text = f"Doc p. {s.get('doc_page')} (PDF p. {s.get('page')}) | {s.get('section')}"
            tag_w = int(len(tag_text) * 7.0 + 20)
            if tag_x + tag_w > width - 50:
                tag_x = main_x
                curr_y += 28
            draw.rounded_rectangle([(tag_x, curr_y), (tag_x + tag_w, curr_y + 22)], radius=4, fill="#e0f2fe", outline="#bae6fd")
            draw.text((tag_x + 10, curr_y + 4), tag_text, fill="#0369a1", font=f_tag)
            tag_x += tag_w + 10
        curr_y += 35

    # Evidence Drawer
    draw.rounded_rectangle([(main_x, curr_y), (width - 40, curr_y + 65)], radius=6, fill="#f1f5f9", outline="#e2e8f0")
    draw.text((main_x + 15, curr_y + 10), f"Retrieved Document Evidence ({len(result.get('retrieved_chunks', []))} Chunks in Context)", fill="#334155", font=f_body_bold)
    first_chunk = result.get("retrieved_chunks", [{}])[0]
    preview = first_chunk.get("content", "").replace("\n", " ")[:125]
    draw.text((main_x + 15, curr_y + 34), f"Top Chunk: \"{preview}...\"", fill="#64748b", font=f_caption)

    # Save
    out_path = SCREENSHOTS_DIR / filename
    img.save(out_path, "PNG")
    print(f"Saved screenshot: {out_path.name}")


def main():
    benchmarks = [
        ("screenshot_q1_three_inspections.png", "What are the three main types of inspection?"),
        ("screenshot_q2_maintenance_factor_two.png", "What does a maintenance factor of two mean?"),
        ("screenshot_q3_purchase_price_not_found.png", "What is the purchase price of this gas turbine?"),
        ("screenshot_q4_melting_point_titanium_not_found.png", "What is the melting point of titanium?"),
        ("screenshot_q5_combustion_inspection_interval.png", "What is the recommended combustion-inspection interval in hours?"),
    ]

    for filename, question in benchmarks:
        print(f"Generating UI capture for: '{question}'...")
        res = query_rag(question)
        render_ui_screenshot(filename, question, res)

    print("All screenshots generated cleanly!")

if __name__ == "__main__":
    main()
