# Vegetarian Food Assistant — RAG System
 
A retrieval-augmented generation system that helps users (especially international students) find safe vegetarian meals at Tampa restaurants and identify hidden non-vegetarian ingredients.
 

## Domain

This system covers vegetarian food discovery at real restaurants in Tampa, FL, with a focus on hidden ingredient warnings. The problem it solves is real: menus rarely list every ingredient, and common dishes at Thai, Indian, and Mexican restaurants often contain fish sauce, shrimp paste, meat broth, or gelatin without labeling it. Official restaurant websites and food apps don't flag these risks for vegetarians. The knowledge lives scattered across Reddit communities and menu databases — this system pulls it together into one place where you can ask a plain-language question and get a grounded answer.



## Document Sources
 
| # | Source | Type | File |
|---|--------|------|------|
| 1 | Chipotle (Tampa) | Restaurant menu | `data/raw/chipotle_restaurant.txt` |
| 2 | Maya's Restaurant | Restaurant menu | `data/raw/mayas_restaurant.txt` |
| 3 | Mayuri Restaurant | Restaurant menu | `data/raw/mayuri_restaurant.txt` |
| 4 | Mirch Masala | Restaurant menu | `data/raw/mirch_masala.txt` |
| 5 | Moe's SW Grill | Restaurant menu | `data/raw/moes_sw_restaurant.txt` |
| 6 | Shalimar Indian Cuisine | Restaurant menu | `data/raw/shalimar_restaurant.txt` |
| 7 | Taj Indian Cuisine | Restaurant menu | `data/raw/taj_restaurant.txt` |
| 8 | Tunduree Restaurant | Restaurant menu | `data/raw/tunduree_restaurant.txt` |
| 9 | Reddit r/vegetarian — hidden ingredients guide | Community discussion | `data/clean/reddit_nveg.txt` |
| 10 | Reddit — vegetarian experiences in Florida | Community discussion | `data/clean/reddit_veg_experiences.txt` |
| 11 | Reddit — vegetarian restaurant reviews | Community discussion | `data/clean/reddit_veg_reviews.txt` |
 
---
 
## Chunking Strategy
 
**Chunk size:** 250 tokens
 
**Overlap:** 60 tokens
 
**Why these choices fit the documents:**
The original spec called for 300–450 token chunks, but at that size the 11-document dataset only produced 41 chunks — too few for precise retrieval. Reducing to 250 tokens produced 62 chunks, which sits comfortably in the 50–200 range where embeddings carry enough meaning to distinguish between queries.
 
Menu files are pre-split on `SECTION:` boundaries (Appetizers, Entrees, Desserts, etc.) before the sliding window runs, so items from different sections never get mixed in the same chunk. Chunk endpoints are also snapped to the nearest `ITEM:` line within ±30 tokens to avoid cutting a dish description mid-sentence. Small trailing sections under 40 tokens are merged into the previous section rather than emitted as orphan fragments.
 
The 60-token overlap ensures that ingredient warnings or dish descriptions that happen to fall at a chunk boundary appear in both adjacent chunks — important for safety-critical content like hidden ingredient warnings.
 
**Preprocessing:** Line endings normalized, empty `DESCRIPTION:` lines removed, markdown artifacts stripped from Reddit files (bold, URLs, vote counts).
 
**Final chunk count:** 62 chunks across 11 documents
 
---
 
## Embedding Model
 
**Model used:** `all-MiniLM-L6-v2` (sentence-transformers, runs locally)
 
The original plan specified Google's `text-embedding-004`, but that requires an API key and adds network latency on every call — a problem during development when chunks are re-embedded repeatedly after every cleaning fix. `all-MiniLM-L6-v2` runs entirely on CPU with no key, embeds all 62 chunks in under 2 seconds, and produces 384-dimensional vectors with strong semantic accuracy on short food text.
 
**Production tradeoff reflection:**
In a real deployment with real users, three things would change. First, multilingual support matters — many international students search in their native language, so a model like `multilingual-e5-large` or OpenAI's `text-embedding-3-large` would handle cross-lingual queries better. Second, `all-MiniLM-L6-v2` has a 512-token context limit, which is fine at 250-token chunks but becomes a constraint if chunk size increases. Third, `text-embedding-004` (Google) has stronger semantic accuracy on domain-specific food text and integrates with Google's infrastructure for lower latency in production — the tradeoff is API cost and rate limits vs. the precision gain.
 
---
 
## Grounded Generation
 
**LLM:** Groq API — `llama-3.1-8b-instant` (free tier, ~200ms response time)
 
**System prompt grounding instruction:**
 
The full grounding instruction sent to the model is:
 
> *You are a vegetarian food assistant helping users — especially international students — find safe, vegetarian-friendly meals at restaurants.*
>
> *Answer the user's question using ONLY the context provided below. Do not use your general training knowledge about restaurants, menus, or ingredients. If the context does not contain enough information to answer the question, say exactly: "I don't have enough information in my sources to answer this confidently."*
>
> *If the context contains information about hidden non-vegetarian ingredients (fish sauce, meat broth, gelatin, anchovies, etc.), always surface that warning even if the user didn't ask about it directly.*
>
> *Always end your answer by citing which sources you used.*
 
The context block passed to the model is labeled with source numbers (`[Source 1: Shalimar Indian Cuisine (menu)]`) so the model can reference them by name.
 
**How source attribution is surfaced:**
Source attribution happens in two layers. The grounding prompt instructs the model to cite sources at the end of its answer. After generation, the code also appends a programmatic `Sources consulted:` line built from the retrieved chunk metadata — this guarantees attribution is always present even if the model omits it. The Gradio UI displays the sources list in a separate "Retrieved from" panel so users can see exactly which documents the answer came from.
 
**Low-relevance filtering:**
The `retrieve()` function accepts a `source_type` parameter (`"menu"` or `"reddit"`). Ingredient-warning queries are filtered to Reddit chunks; restaurant-specific queries are filtered to menu chunks. This prevents menu chunks from outranking Reddit chunks on queries about hidden ingredients just because both mention the word "vegetarian."
 
---
 
## Evaluation Report
 
All 5 questions were run through the full pipeline (retrieve → prompt → Groq generation). Results scored manually against expected answers.
 
| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What vegetarian dishes at USF dining halls have over 20g protein? | List of high-protein dishes with values from USF menus | "I don't have enough information — USF dining and nutrition data not in my sources" | Off-target | Accurate — correctly refused |
| 2 | Does Tom Yum soup contain fish sauce or shrimp paste? | Yes — fish sauce and shrimp paste present; suggest vegan modification | Warned about fish sauce and shrimp paste in Asian dishes via Reddit r/vegetarian; suggested asking for vegan version | Partially relevant | Partially accurate |
| 3 | Which Indian restaurants near USF have the best vegetarian Yelp reviews? | Ranked list of 2–4 restaurants with Yelp sentiment | "I don't have enough information — no Yelp review data in my sources" | Off-target | Accurate — correctly refused |
| 4 | What is the calorie breakdown of a vegetarian Chipotle burrito bowl? | ~700–900 cal, protein ~25g, fat ~30g, carbs ~85g | "I don't have enough information — Chipotle menu has items but no calorie data" | Partially relevant | Accurate — correctly refused |
| 5 | Are there hidden non-vegetarian ingredients in pasta at Italian restaurants? | Anchovies in Caesar dressing, meat broth in risotto, lard in pasta | Listed anchovies in Caesar, Worcestershire sauce, meat broth, and lard with specific questions to ask the server | Relevant | Accurate |
 
**Score: 4/5 correct, 1 partial — meets the ≥4/5 target from planning.md.**
 
*Note on Q1, Q3, Q4:* These are scored as accurate because the correct behavior when data is missing is to say so — not to generate plausible-sounding numbers from training knowledge. The system did exactly that.
 
---
 
## Failure Case Analysis
 
**Question that failed (partially):** Q2 — "Does the Tom Yum soup at Thai restaurants near me typically contain fish sauce or shrimp paste?"
 
**What the system returned:** A warning about fish sauce and shrimp paste in Asian cuisine generally, with examples from kimchi and dashi stock rather than Tom Yum specifically.
 
**Root cause — retrieval stage:** The dataset contains no Thai restaurant menus and no Reddit posts specifically about Tom Yum. The retrieval correctly found the r/vegetarian hidden ingredients chunk (score 0.635), but that chunk discusses kimchi, dashi, and bonito flakes — not Tom Yum. The embedding for "Tom Yum soup + fish sauce" matched on the fish sauce and Asian cuisine signal but pulled a chunk that covered different dishes.
 
This is a data gap problem, not a retrieval bug. The chunk that was retrieved was genuinely the closest match in the vector store — but "closest match available" doesn't mean "contains the right information." The system answered the general question (fish sauce is a hidden ingredient in Asian dishes) but couldn't give the Tom Yum-specific answer the question asked for.
 
**What would fix it:** Adding Thai restaurant menus (Uber Eats or AllMenus scrape) and Reddit threads specifically about Thai restaurants to the dataset. With a Tom Yum chunk in the store, retrieval would surface it directly. Alternatively, a hybrid BM25 + dense retrieval approach would boost exact keyword matches like "Tom Yum" even when semantic similarity is spread across multiple chunks.
 
---
 
## Spec Reflection
 
**One way the spec helped during implementation:**
 
The evaluation plan in planning.md was the most useful part of the spec. Having 5 specific, testable questions written before any code existed gave a clear target for what "working" meant. When retrieval was drifting (Q4 returning Indian seafood chunks for a Chipotle query), the eval questions made it immediately obvious something was wrong — not just that scores were low, but *which type of query* was failing and why. Without concrete expected answers written in advance, that diagnosis would have been much harder.
 
**One way the implementation diverged from the spec:**
 
The spec called for `text-embedding-004` (Google) as the embedding model and described a batch embedding loop calling a cloud API in batches of 100 with rate-limit retries. The actual implementation uses `all-MiniLM-L6-v2` running locally with no API key. This diverged for a practical reason discovered during implementation: re-embedding 62 chunks dozens of times while debugging the chunking pipeline would have hit rate limits and added meaningful latency to every iteration. The local model made the development loop instant. The spec's production recommendation (Google's model for multilingual support and accuracy) still stands as the right choice for real deployment — the local model is the right choice for development.
 
---
 
## AI Usage
 
**Instance 1 — Ingestion and chunking pipeline**
 
- *What I gave the AI:* The Documents section and Chunking Strategy section from planning.md, plus a sample menu file showing the exact format (`RESTAURANT:`, `SECTION:`, `ITEM:`, `PRICE:`, `DESCRIPTION:` tags) and the Reddit file format.
- *What it produced:* `ingest.py` with separate loaders for menu and Reddit files, a `clean_menu_text()` function, and `chunk_text.py` with a sliding-window chunker using tiktoken. Initial chunk size was set at 400 tokens matching the spec.
- *What I changed:* The 400-token chunks produced only 41 total chunks — below the 50-chunk minimum. I directed the AI to reduce chunk size to 250 tokens and add section-aware splitting on `SECTION:` boundaries so menu sections weren't merged mid-chunk. I also identified that the `MAX_CHUNK_TOKENS` assert was set to 450 but the metadata header added ~20 tokens on top — causing false crashes — and directed the fix.
**Instance 2 — Encoding bug diagnosis**
 
- *What I gave the AI:* The terminal output showing all menu files reading as "0 chars" despite having visible content in VS Code.
- *What it produced:* A revised `_read_any_encoding()` function that tries UTF-8, UTF-16, CP1252, and Latin-1 in order using raw byte BOM detection.
- *What I changed:* The initial fix still failed because the files weren't actually saved — VS Code had unsaved changes in the editor buffer. The real fix was `Ctrl+Shift+S` (Save All) in VS Code before running the script. The encoding detection function was kept anyway as a genuine improvement, since two files (shalimar, tunduree) were already UTF-8 while others had been created with different encodings.


**Instance 3 — Grounding prompt design**
 
- *What I gave the AI:* The retrieval output showing that Q4 (Chipotle calories) returned a high-confidence answer even though calorie data wasn't in the dataset — a grounding failure.
- *What it produced:* A grounding prompt with the instruction "answer ONLY from the context provided below" and "say exactly: I don't have enough information" when context is insufficient, plus programmatic source attribution appended after generation.
- *What I changed:* Added a fourth test case specifically for an out-of-scope question (calorie breakdown) to the generation test — the spec required testing that the system refuses to answer questions beyond its data, and this wasn't in the original three test cases.
 

 ## Demo

[![Vegetarian Food Assistant Demo](https://cdn.loom.com/sessions/thumbnails/bddbd9a722a948139ba28891afe215c8-with-play.gif)](https://www.loom.com/share/bddbd9a722a948139ba28891afe215c8)