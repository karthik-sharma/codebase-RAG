# RAG Internals — The Concepts Behind This Repo

*This document is different from `ARCHITECTURE.md`. `ARCHITECTURE.md` is a map of the
code — what file does what, how to run it. This document is a **RAG theory
reference** — for each concept, it explains the general idea first (as if this repo
didn't exist), then points at exactly where and how this repo implements it, and
finally covers the RAG concepts this repo does **not** implement yet, explained the
same way, so the gap is understandable and not just a checklist item.*

## Table of contents

- [Part 1 — Why RAG exists](#part-1--why-rag-exists)
- [Part 2 — Concepts this repo implements](#part-2--concepts-this-repo-implements)
  1. [Embeddings & semantic search](#1-embeddings--semantic-search)
  2. [Lexical search (BM25)](#2-lexical-search-bm25)
  3. [Hybrid search & Reciprocal Rank Fusion](#3-hybrid-search--reciprocal-rank-fusion)
  4. [Chunking strategy](#4-chunking-strategy)
  5. [Reranking (cross-encoders)](#5-reranking-cross-encoders)
  6. [Conversational retrieval / query rewriting](#6-conversational-retrieval--query-rewriting)
  7. [Grounded generation](#7-grounded-generation)
  8. [Contextual chunk enrichment](#8-contextual-chunk-enrichment)
  9. [Index idempotency & stable IDs](#9-index-idempotency--stable-ids)
  10. [Ingestion-time data governance](#10-ingestion-time-data-governance)
  11. [Incremental indexing](#11-incremental-indexing)
  12. [Retrieval & generation evaluation](#12-retrieval--generation-evaluation)
- [Part 3 — Concepts this repo does NOT implement yet](#part-3--concepts-this-repo-does-not-implement-yet)
  1. [Multi-tenant / multi-project indexing](#1-multi-tenant--multi-project-indexing)
  2. [Observability / tracing](#2-observability--tracing)
  3. [Hallucination / faithfulness guardrails (at answer time)](#3-hallucination--faithfulness-guardrails-at-answer-time)
  4. [Broader language coverage for chunking](#4-broader-language-coverage-for-chunking)
  5. [Productization (CLI)](#5-productization-cli)
  6. [The generator model itself is a ceiling](#6-the-generator-model-itself-is-a-ceiling)
- [Quick reference table](#quick-reference-table)

---

## Part 1 — Why RAG exists

An LLM's knowledge is frozen at training time, and it has never seen your private code.
There are three ways to give it access to information it doesn't already know:

1. **Fine-tuning** — retrain (part of) the model on your data. Expensive, goes stale the
   moment your code changes again, and the model still can't tell you *which file* an
   answer came from — it just "knows" it now, the same way it knows anything else.
2. **Stuff everything into the context window** — paste the whole codebase into the
   prompt every time. Works for genuinely small codebases, but gets expensive fast (every
   token costs time and, on hosted models, money) and LLMs get measurably worse at using
   information buried in the middle of a very long context (the "lost in the middle"
   effect) — bigger context isn't free accuracy.
3. **Retrieval-Augmented Generation (RAG)** — keep the data outside the model entirely,
   in a searchable index. At question time, search for just the relevant pieces, and
   only put *those* in the prompt. Cheap per query, trivially updated (re-index when code
   changes), and naturally supports citing sources, because you know exactly which chunk
   the answer came from.

This repo is entirely the third option. Everything below is really answering one
question, over and over, at a different layer each time: **"how do we find the right
handful of chunks, and how do we get a model to answer honestly using them?"**

---

## Part 2 — Concepts this repo implements

### 1. Embeddings & semantic search

**The concept**: an embedding model is a neural network trained so that text with
similar *meaning* maps to nearby points in a high-dimensional vector space — e.g.
"login" and "authenticate" end up close together even though they share no letters.
Once every chunk is embedded once (offline), answering "what's relevant to this
question" becomes a geometry problem: embed the question the same way, then find which
stored vectors are closest to it (typically by cosine similarity — the angle between two
vectors, ignoring their length).

Comparing a query vector against *every* stored vector one by one (brute force) doesn't
scale past a small number of chunks. Real vector databases use **approximate nearest
neighbor (ANN)** search — Chroma, like most, builds an HNSW graph (Hierarchical
Navigable Small World) under the hood, a structure that lets you find "close enough"
neighbors in roughly logarithmic time instead of scanning everything. This repo never
touches that layer directly — it's what makes `vector_store.similarity_search_with_score`
fast — but it's worth knowing it's there, since it's the whole reason vector databases
exist as a distinct category of database.

**In this repo**: `nomic-embed-text`, run locally via Ollama (`OllamaEmbeddings` in
`app/rag/store.py`), embeds both every chunk (at ingestion) and every query (at
retrieval). Chroma is the vector database (`chroma_db/` on disk).

### 2. Lexical search (BM25)

**The concept**: embeddings are good at *meaning*, bad at *exact terms* — a specific
function name, an exact error string, a file path fragment might not be the closest
semantic match to anything, but it's obviously what you're looking for if it appears
verbatim. **BM25** is a decades-old scoring algorithm built for exactly this: it scores
a document highly for a query if it contains the query's words, weighted by:
- **term frequency** — how often each query word appears in that document,
- **inverse document frequency** — rare words across the whole corpus count for more
  than common ones (a document matching "expense" doesn't stand out if "expense" is in
  every document; matching "register_user" is far more distinguishing),
- **length normalization** — a long document shouldn't win purely by containing more
  words overall.

**In this repo**: `BM25Retriever` (from `langchain-community`, backed by the
`rank-bm25` package), built fresh from `chunks.json` every time `hybrid_search.py` is
imported. This is the "keyword search" half of hybrid search.

### 3. Hybrid search & Reciprocal Rank Fusion

**The concept**: semantic search and lexical search each cover the other's blind spot,
so real systems combine both. The tricky part is *how* to combine them — a BM25 score
and a cosine-similarity-derived distance are not on the same scale, so you can't just
add or average them meaningfully. **Reciprocal Rank Fusion (RRF)** sidesteps this by
ignoring raw scores entirely and using each result's **rank position** in each list
instead:

```
score(doc) = Σ over each retriever r :  weight_r / (k + rank_r(doc))
```

where `rank_r(doc)` is that document's position in retriever `r`'s ranked list (1st,
2nd, 3rd...), and `k` is a constant (commonly 60) that softens how much the very top
rank dominates. A document ranked highly by *either* retriever gets a rank-based boost,
without ever needing to compare incomparable raw scores.

**In this repo**: `EnsembleRetriever` (`langchain-classic`) combines the vector
retriever and `BM25Retriever` this way, weighted `[0.7, 0.3]` in favor of semantic
search, merging results by a shared `chunk_id` (see [§9](#9-index-idempotency--stable-ids))
rather than the library's default of matching on exact `page_content` text, which is
more fragile.

**A known gap, found via the eval harness ([§12](#12-retrieval--generation-evaluation))**:
this stage — and the reranker after it — can still be fooled when a *concept* in the
question (e.g. "login") is also, coincidentally, close to unrelated file/component names
(`Login.jsx`, `Login.css`). Hybrid search and RRF fuse two *retrieval* signals; neither
one disambiguates "the behavior of logging in" from "things literally named Login." Not
yet fixed.

### 4. Chunking strategy

**The concept**: neither embedding models nor LLM prompts work well on whole files —
too large, too unfocused. Content has to be split into **chunks** first. The naive
approach is fixed-size character/token splitting; the problem is it's blind to
structure and will cut a function in half exactly as readily as it'll cut between two
functions. A better approach for source code is **structure-aware chunking** — parsing
the code into its real syntax tree and splitting at actual boundaries (a whole function,
a whole class), so every chunk is something a human would also consider "one unit."

Chunk size itself is a real tradeoff even with structure-aware chunking: too small and a
chunk loses the surrounding context needed to make sense of it in isolation; too big and
it dilutes the specific, relevant part of a match with irrelevant surrounding text
(wasting prompt space that could hold another, different relevant chunk instead).

**In this repo**: Python is parsed with the standard library's `ast` module;
JS/TS/JSX/TSX with **tree-sitter** (a real parser-generator with grammars for many
languages, via `tree-sitter-language-pack`) — both split at function/class/method
boundaries, plus one grouped chunk per file for its imports. Everything else falls back
to a generic `RecursiveCharacterTextSplitter` (character-based, 1000 chars, 200 overlap)
— acceptable for prose/config where there's no equivalent notion of "a function" to
preserve.

### 5. Reranking (cross-encoders)

**The concept**: there are two different neural architectures for scoring
query-vs-document relevance, with a real speed/accuracy tradeoff:
- A **bi-encoder** (what embedding models are) encodes the query and each document
  *separately* into vectors, then compares the vectors. Because documents are encoded
  independently of any particular query, they can all be embedded once, offline, ahead
  of time — that's what makes a searchable index possible at all.
- A **cross-encoder** feeds the query *and* a candidate document into the model
  *together*, letting the model directly attend across both texts at once, and outputs
  a single relevance score. This is meaningfully more accurate (it can actually reason
  about how the two relate), but it can't be precomputed — every candidate has to be
  scored fresh, at query time, which is far too slow to run against an entire corpus.

The standard resolution is **retrieve-then-rerank**: use the cheap bi-encoder/lexical
search to narrow a large corpus down to a small candidate pool, then spend the
cross-encoder's expensive-but-accurate scoring only on that shortlist.

**In this repo**: hybrid search (§3) pulls `CANDIDATE_POOL_SIZE = 10` candidates; a
cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, via `HuggingFaceCrossEncoder` +
`CrossEncoderReranker`, wired through `ContextualCompressionRetriever`) rescores those
10 against the real question text and keeps only the final `top_k` (3, by default).

### 6. Conversational retrieval / query rewriting

**The concept**: a follow-up question in a conversation is often not self-contained —
*"what about the login one?"* relies entirely on a prior turn to mean anything. If you
embed and search using that raw text, retrieval fails, not because the retriever is bad,
but because the query itself doesn't carry enough information. The fix is a **query
rewriting** (sometimes called "condensation") step: before retrieval, an LLM call takes
the conversation history plus the new question and rewrites it into a standalone form
that means the same thing without needing the history.

**In this repo**: `rewrite_standalone_question` in `generate.py` — skipped on the first
question (nothing to rewrite against yet), otherwise a `ChatOllama` call producing the
standalone version, which is what actually gets passed to `hybrid_search`. The user's
*original* wording is still what the final answer-generation prompt sees, so the answer
still speaks to what was literally asked.

### 7. Grounded generation

**The concept**: **grounding** means constraining the model to answer only from
provided context, rather than from whatever it happens to remember from training — the
whole point of building a retrieval pipeline is undermined if the model ignores it and
answers from memory anyway. This is done through prompt instructions ("answer only using
the provided context," "cite the source for every fact") — necessary, but *not*
sufficient on its own, since nothing enforces the model actually follows those
instructions (see [§3 in Part 3](#3-hallucination--faithfulness-guardrails-at-answer-time)).

**In this repo**: the prompt built in `answer_question` (`generate.py`) requires: answer
only from context, cite the source file for *every* fact used (explicitly not just one,
if multiple files contributed — this was tightened after observing the model drop
citations inconsistently), include function/class name and line numbers when available,
and don't invent information.

### 8. Contextual chunk enrichment

**The concept**: a chunk embedded in complete isolation can lose meaning it would have
had in its original context — an import list or a short function reads differently
without knowing which file/module it belongs to. One mitigation, used by several
production RAG systems, is to enrich the text *before embedding* with a small amount of
surrounding context (e.g. its file path, or a one-line summary of the containing
document) — cheap to do, and it means the semantic representation itself carries that
context, not just metadata sitting next to it that only the eventual reader (not the
retriever) benefits from.

**In this repo**: `chunk_documents.py` prefixes every embedded chunk with
`File: {path}\n\n` before the actual code. This is also what makes a query like "what do
we import in **backend** files" actually favor backend files *semantically* — without
it, the embedding has zero notion of directory structure, only code content.

### 9. Index idempotency & stable IDs

**The concept**: adding documents to a vector store is not automatically an "upsert" —
without an explicit, stable identifier per document, re-adding the same content twice
just creates two separate entries. Any system that expects to be re-indexed (which is
almost all of them — code changes) needs either (a) a full-rebuild strategy (wipe
everything, re-add from scratch) or (b) stable IDs so a second add can correctly replace
the first instead of duplicating it. Stable IDs are also what let results computed by
*different* retrieval methods (lexical, semantic) be recognized as "the same underlying
item" when merging their rankings — and, further, what makes *targeted* updates
(§11 — re-add just one file's chunks) possible at all.

A sharp edge worth knowing generally, not just here: passing a stable ID string in your
own application code is not the same as that string actually becoming the database's
real internal ID — most vector stores (Chroma included) will silently auto-generate
their own ID unless you explicitly tell the client "use *this* ID," at which point any
ID-based delete/update you attempt later using your own scheme will quietly match
nothing.

**In this repo**: strategy (b) — a `chunk_id` (`file:chunk_index`, where `chunk_index`
is scoped to that file, not a global position — see §11) is computed once in
`chunk_documents.py` and passed explicitly as `ids=` on every `vector_store.add_documents()`
call, so it becomes Chroma's actual internal document ID, not just applicaton-level
metadata. The same `chunk_id` is reused identically in `chunks.json` and the BM25 index
— this is what lets `EnsembleRetriever` correctly merge a BM25 hit and a semantic hit
for the same underlying chunk (§3), and what lets ingestion delete exactly the right
chunks when a file changes (§11).

### 10. Ingestion-time data governance

**The concept**: any pipeline that scans arbitrary files and embeds their content needs
to treat "what should never be indexed" as a first-class concern, not an afterthought —
embeddings and stored chunk text are not access-controlled or encrypted by default, so a
secret that gets embedded is a secret that's now sitting in the vector store (and
whatever debug files the pipeline writes) in a form outside its owner's original access
controls.

**In this repo**: `scan_project.py` respects every `.gitignore` actually present in the
target project (not just a hand-maintained guess at what's unsafe), using real gitignore
pattern semantics via the `pathspec` library — see `ARCHITECTURE.md §5.1` for the full
mechanism, including the nested-`.gitignore` case that a simpler "check one root file"
approach would have missed.

### 11. Incremental indexing

**The concept**: re-embedding an entire corpus from scratch on every update doesn't
scale — the standard approach is to detect *what changed* since the last index build
(via a content hash or modification timestamp per file), and only re-chunk/re-embed/
upsert those changed files, while also detecting and removing chunks for files that were
deleted entirely. This requires stable, targeted-deletable chunk IDs (§9) as a genuine
prerequisite, not just a nice-to-have — you can't correctly "replace just this file's
chunks" without a reliable way to identify exactly which stored entries belong to that
file and only that file.

**In this repo**: a manifest (`index_manifest.json`, `{file_path: content_hash}`,
SHA-256) is written after every ingestion run. The next run hashes every current file
and compares: an unchanged file's previously-computed chunks are reused outright (no
re-chunking, no re-embedding); a new or changed file is re-chunked, its old chunk IDs
(if any) are deleted from Chroma, and its fresh chunks are added; a file that's vanished
since the last run has its old chunk IDs deleted and is dropped from the manifest. The
very first run (no manifest yet) is always a full scan + full rebuild — both a
legitimate bootstrap case and, in practice, an automatic migration path whenever the
chunk-ID scheme itself changes underneath (which happened once already — see
`ARCHITECTURE.md §6.4`).

**A real, two-part bug found while building this** (worth knowing as a case study in
why "add incremental logic" is rarely just adding a diff check): first, the existing
`chunk_id` scheme turned out to be a *global* list position, not scoped per file — stable
under a full-rebuild-always strategy, but silently unstable the moment one file's chunk
count could change independently of others. Second, and more subtly, `chunk_id` had
never actually been passed to Chroma as the real document ID at all (§9's "sharp edge") —
so even after fixing the ID scheme, targeted deletes matched nothing until `ids=` was
passed explicitly on every add. Neither of these was visible under the old
full-wipe-and-rebuild strategy; both were only discoverable by actually building and
testing the incremental path, not by inspecting the old code for correctness.

### 12. Retrieval & generation evaluation

**The concept**: "does this look right" on one or two manually-typed queries is not
measurement — it can't detect a regression introduced by a later change, and it doesn't
generalize past whatever you happened to type. A real evaluation setup has two halves:

- **Retrieval metrics** — given a fixed set of `(question, correct chunk/file)` pairs,
  compute things like **Precision@k** (of the top k retrieved, how many are actually
  relevant), **Recall@k** (of all truly relevant chunks, how many made it into the top
  k), and **MRR** (Mean Reciprocal Rank — how high up the *first* correct result appears,
  averaged across many questions). (A fuller framework would also compute **NDCG**,
  Normalized Discounted Cumulative Gain, which extends this to *graded* relevance — some
  results more relevant than others rather than a binary yes/no — but that needs richer,
  hand-labeled test data than a first version needs to be useful.)
- **Generation metrics** — given the retrieved context and the model's answer,
  **faithfulness** (is every claim in the answer actually supported by the provided
  context, or invented) and **answer relevance** (does the answer actually address the
  question asked). Frameworks like RAGAS formalize exactly this pair of measurements,
  typically via an LLM acting as an automated judge.

**In this repo**: `app/rag/eval.py` — a fixed `EVAL_CASES` list of
`(question, expected_files)` pairs run against `hybrid_search()` directly, scored for
**Hit Rate**, **Precision@k**, **Recall@k**, and **MRR**; plus a **faithfulness**
check that runs the real `answer_question()` from `generate.py` and then makes a
*separate* LLM call asking whether the answer's claims are actually supported by the
retrieved context.

**Why this immediately paid for itself**: the very first real run surfaced two genuine
findings a single manual query would very plausibly have missed — a retrieval confusion
between the *concept* "login" and the *literal filenames* `Login.jsx`/`Login.css` (§3's
known gap), and evidence that the faithfulness judge itself isn't fully reliable at this
model's scale (it once flagged an answer as unsupported by claiming a word was missing
from context that was, on inspection, actually present). Both are now known, recorded
limitations instead of invisible ones — which is exactly what an eval harness is for:
not fixing quality, but making its actual state visible and re-checkable.

---

## Part 3 — Concepts this repo does NOT implement yet

Each of these is a real, standard part of a production RAG system. They're explained the
same way as above — concept first, then how it would apply here — specifically so the
gap is legible, not just a name on a list.

### 1. Multi-tenant / multi-project indexing

**The concept**: a RAG system meant to serve more than one "corpus" (here: more than one
codebase) needs isolation between them — either separate vector-store collections per
tenant/project, or one collection with a metadata field (e.g. `project_id`) that every
query filters on. Without this, ingesting a second corpus either mixes its data with the
first (polluting results) or, worse, silently destroys the first.

**Why it matters here**: there is currently exactly one Chroma collection
(`"codebase"`). Incremental indexing (§11 in Part 2) means a same-project re-ingest no
longer wipes anything unnecessarily, but ingesting a genuinely *different* project still
either mixes its chunks into the same collection as whatever was ingested before, or (on
a bootstrap run, if the manifest happens to be absent) wipes it outright. This is the
most structurally important gap left to close before this tool could reasonably be used
across more than one project without deliberately managing that overlap yourself.

### 2. Observability / tracing

**The concept**: production RAG systems log every query's full trace — what was
retrieved (and each candidate's scores), what the final assembled prompt looked like,
and what the model answered — so a bad answer can be debugged *after the fact* from the
log, instead of only being diagnosable by reproducing it live. This logged history is
also directly where a real evaluation set (§12 in Part 2) usually comes from in a mature
system: real queries a user actually asked, with their outcomes, curated into test cases
over time. Hosted tools like LangSmith/Langfuse do this for you; a local-first
equivalent would just be writing each trace to a local SQLite file or JSONL log.

**Why it matters here**: every debugging session in this project so far has required
re-running a query live to see what happened — there's no record of any past query,
what was retrieved for it, or what was answered. The eval set in §12 was hand-written
for exactly this reason — there was no logged history to mine it from.

### 3. Hallucination / faithfulness guardrails (at answer time)

**The concept**: telling a model "only use the provided context" (§7 in Part 2) is a
request, not an enforcement mechanism — nothing stops the model from ignoring it. A
faithfulness guardrail is a *separate*, automated check run *after* generation: either
an NLI-style (natural language inference) model checking whether each sentence in the
answer is logically entailed by the retrieved context, or a second LLM call acting as a
judge. `eval.py` (§12 in Part 2) already builds this exact mechanism — but only as an
*offline* scoring signal, run manually against a fixed test set, not as a live check on
every real answer a user actually sees.

**Why it matters here**: nothing currently verifies, during an actual `generate.py`
session, that an answer stayed grounded in what was retrieved — the faithfulness
checking code exists, but only runs when `eval.py` is run deliberately, not on real
usage. Wiring the same check into `answer_question()` itself (with, e.g., a bounded
retry on failure) is the natural next step — and would also be a good first real use
case for LangGraph, since "generate → check → conditionally retry" is a loop, not a
straight-line chain.

### 4. Broader language coverage for chunking

**The concept**: structure-aware chunking (§4 in Part 2) is only as broad as the parsers
wired up for it. `tree-sitter-language-pack` (already a dependency here) ships grammars
for dozens of languages — extending chunking to, say, Go or Rust is mechanically the
same pattern already built for JS/TS (pick the grammar, walk the top-level nodes, map
node types to chunk types), not a new architectural problem.

**Why it matters here**: any language other than Python/JS/TS/JSX/TSX currently falls
back to the generic character splitter (§4), with the same function-splitting risk that
motivated building AST-based chunking in the first place.

### 5. Productization (CLI)

**The concept**: not a RAG concept specifically, but a systems one — a tool meant to be
reused needs its varying inputs (which project to index, what question to ask) as
runtime arguments, not values edited into source code before every run.

**Why it matters here**: `PROJECT_DIR` in `ingestion.py` is still a hardcoded path.

### 6. The generator model itself is a ceiling

**The concept**: retrieval quality has diminishing returns the moment the *generation*
model is the weaker link — perfect context handed to a model that reasons poorly over
it still produces a mediocre answer. This isn't something retrieval-side engineering can
ever fix; it's a separate, orthogonal axis of quality (which model actually synthesizes
the final answer) — and it applies just as much to a model *judging* an answer as to the
model that *wrote* it.

**Why it matters here**: `llama3.2` run locally via Ollama is a small model relative to
what a hosted/production system would typically use for final answer synthesis. Running
fully local was a deliberate tradeoff in this project (no API keys, no data leaving the
machine) — worth naming as a real limit on overall answer quality, not a bug to fix
within the retrieval pipeline. It's also, concretely, the reason the eval harness's own
faithfulness judge (§12 in Part 2) isn't fully trustworthy yet — the same small model is
being asked to grade its own kind of output.

---

## Quick reference table

| Concept | Status | Where |
|---|---|---|
| Embeddings & semantic search | ✅ Implemented | `app/rag/store.py` |
| Lexical search (BM25) | ✅ Implemented | `app/rag/searching_strategy/hybrid_search.py` |
| Hybrid search (RRF) | ✅ Implemented | `hybrid_search.py` (`EnsembleRetriever`) |
| AST-aware chunking | ✅ Implemented (Python, JS/TS) | `chunking_strategy/chunk_python_file.py`, `chunk_js_ts_file.py` |
| Reranking (cross-encoder) | ✅ Implemented | `hybrid_search.py` (`CrossEncoderReranker`) |
| Conversational query rewriting | ✅ Implemented | `app/rag/generate.py` |
| Grounded generation / citation | ✅ Implemented | `generate.py` prompt |
| Contextual chunk enrichment | ✅ Implemented | `chunk_documents.py` |
| Index idempotency & stable IDs | ✅ Implemented | `ingestion.py`, `chunk_documents.py` |
| Ingestion-time data governance | ✅ Implemented | `scan_project.py` |
| Incremental indexing | ✅ Implemented | `ingestion.py`, `scan_project.py`, `manifest.py` |
| Retrieval & generation evaluation | ✅ Implemented | `app/rag/eval.py` |
| Multi-tenant / multi-project indexing | ❌ Not built | — |
| Observability / tracing | ❌ Not built | — |
| Hallucination guardrail at answer time | ❌ Not built (offline version exists in `eval.py`) | — |
| Broader language chunking coverage | ❌ Not built | — |
| CLI / productization | ❌ Not built | — |
| Frontier-scale generator model | ❌ Deliberately not used (local-first tradeoff) | — |
