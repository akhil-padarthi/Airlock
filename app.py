"""Airlock web UI.  Run:  streamlit run app.py   (serves on 127.0.0.1 only)"""

from airlock import airgap

airgap.install()  # before any other import that might open a socket

import streamlit as st  # noqa: E402

from airlock.config import AirlockConfig  # noqa: E402
from airlock.pipeline import Airlock  # noqa: E402

st.set_page_config(page_title="Airlock", page_icon="🔒", layout="wide")


@st.cache_resource(show_spinner="Loading local models...")
def load_components(embed_model: str, llm_model: str, use_reranker: bool):
    """Model handles only (no document data), shared across browser sessions."""
    from airlock.models import build_embedder, build_generator, build_reranker

    cfg = AirlockConfig.from_env(embed_model=embed_model, llm_model=llm_model, use_reranker=use_reranker)
    reranker = build_reranker(cfg) if use_reranker else None
    return build_embedder(cfg), build_generator(cfg), reranker


def purge() -> None:
    session = st.session_state.get("session")
    if session is not None:
        session.close()
    for key in ("session", "report", "messages", "doc_key"):
        st.session_state.pop(key, None)


def render_answer(ans) -> None:
    if ans.answered:
        badge = {"verified": ":green-background[Verified against the document]",
                 "partial": ":orange-background[Partially verified - see flags]",
                 "unchecked": ":gray-background[Unchecked]"}[ans.verification]
    else:
        reason = {"low_confidence": "no confident match in the document",
                  "model_declined": "passages don't contain the answer",
                  "failed_source_validation": "draft answer failed source validation"}.get(ans.reason, ans.reason)
        badge = f":red-background[Not answered - {reason}]"
    st.markdown(badge)
    st.markdown(ans.text)
    for c in ans.sentences:
        if not c.supported:
            st.warning(f"Unverified sentence: \"{c.text}\"  \n{'; '.join(c.reasons)}")
    if ans.sources:
        title = "Sources" if ans.answered else "Closest passages"
        with st.expander(f"{title} ({len(ans.sources)})"):
            for s in ans.sources:
                rerank = f" · rerank relevance {s.rerank_prob:.2f}" if s.rerank_prob is not None else ""
                st.markdown(f"**[{s.label}] {s.pages_label}** · cosine {s.vector_sim:.2f}{rerank}")
                st.caption(s.text[:700] + ("..." if len(s.text) > 700 else ""))
    session = st.session_state.get("session")
    record = session.provenance.get(ans.record_id) if session else None
    if record:
        with st.expander("Provenance record"):
            st.json(record.to_dict(), expanded=False)


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("Airlock")
    st.caption("Air-gapped Q&A over massive PDFs. Nothing leaves this machine; nothing is kept.")

    if airgap.self_test():
        st.success("Air-gap active: outbound connections are blocked", icon="🔒")
    else:
        st.error("Air-gap guard not active")

    base = AirlockConfig.from_env()
    with st.expander("Settings"):
        llm_model = st.text_input("LLM (Ollama)", base.llm_model)
        embed_model = st.text_input("Embeddings (Ollama)", base.embed_model)
        chunking = st.selectbox("Chunking", ["semantic", "fixed"], index=0 if base.chunking == "semantic" else 1)
        use_reranker = st.toggle("Cross-encoder reranking", base.use_reranker)
        top_k = st.slider("Stage 1: vector candidates", 5, 100, base.vector_top_k)
        top_n = st.slider("Stage 2: passages kept", 1, 10, base.rerank_top_n)
        min_rerank = st.slider("Gate: min rerank relevance", 0.0, 1.0, base.min_rerank_prob, 0.05)
        min_sim = st.slider("Gate: min vector similarity", 0.0, 1.0, base.min_vector_sim, 0.05)
        strict = st.toggle("Strict validation (refuse if any sentence unverified)", base.strict_validation)

    cfg = base.with_(llm_model=llm_model, embed_model=embed_model, chunking=chunking,
                     use_reranker=use_reranker, vector_top_k=top_k, rerank_top_n=top_n,
                     min_rerank_prob=min_rerank, min_vector_sim=min_sim, strict_validation=strict)

    uploaded = st.file_uploader("PDF (held in memory, never saved)", type=["pdf"])
    if uploaded is not None:
        doc_key = (uploaded.name, uploaded.size, chunking, embed_model)
        if st.session_state.get("doc_key") != doc_key and st.button("Index document", type="primary"):
            purge()
            try:
                embedder, generator, reranker = load_components(embed_model, llm_model, use_reranker)
                from airlock.models import build_index

                session = Airlock(embedder, generator, reranker, build_index, cfg,
                                  {"embeddings": f"ollama/{embed_model}", "llm": f"ollama/{llm_model}",
                                   "reranker": cfg.reranker_hf_id if use_reranker else None})
                bar = st.progress(0.0, text="Reading PDF...")
                report = session.ingest(
                    uploaded.getvalue(), uploaded.name,
                    progress=lambda stage, done, total: bar.progress(min(1.0, done / max(1, total)),
                                                                     text=f"{stage}: {done:,}/{total:,}"),
                )
                bar.empty()
                st.session_state.update(session=session, report=report, messages=[], doc_key=doc_key)
            except Exception as e:  # noqa: BLE001
                st.error(f"Indexing failed: {e}\n\nRun `airlock doctor` to check Ollama and models.")

    session: Airlock | None = st.session_state.get("session")
    report = st.session_state.get("report")
    if session is not None and report is not None:
        session.config = cfg.with_(chunking=session.config.chunking)  # thresholds apply live
        st.divider()
        st.markdown(f"**{report.document}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pages", f"{report.pages:,}")
        c2.metric("Chunks", f"{report.chunks:,}")
        c3.metric("Index time", f"{report.seconds:.0f}s")
        for w in report.warnings:
            st.warning(w)
        st.download_button("Export provenance log (JSONL)", session.provenance.to_jsonl() or "",
                           file_name="airlock_provenance.jsonl", mime="application/jsonl",
                           disabled=len(session.provenance) == 0)
        if st.button("Purge session", help="Erase the index, chunks, chat and log from memory"):
            purge()
            st.rerun()

    blocked = airgap.status()["blocked_attempts"]
    if blocked:
        st.caption(f"Blocked {len(blocked)} outbound attempt(s): "
                   + ", ".join(sorted({b['host'] for b in blocked})[:5]))


# ------------------------------------------------------------------ main panel

st.header("Ask your document")
session = st.session_state.get("session")
if session is None:
    st.info("Upload a PDF in the sidebar and click **Index document**. Large files (1,000+ pages) take a few "
            "minutes the first time: every sentence is embedded locally to find topic boundaries.")

for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg["answer"])

question = st.chat_input("Ask about the document", disabled=session is None)
if question and session is not None:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching, reranking, answering, verifying..."):
            try:
                answer = session.ask(question)
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed: {e}")
                answer = None
        if answer is not None:
            render_answer(answer)
            st.session_state["messages"].append({"role": "assistant", "answer": answer})
