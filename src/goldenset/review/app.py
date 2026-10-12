"""Streamlit review page. Run with `goldenset review --run runs/<id>`."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from goldenset.models import Record, ReviewDecision
from goldenset.review.store import ReviewStore
from goldenset.stages.enrich import render_conversation
from goldenset.taxonomy import FAILURE_MODES, INTENTS

REJECT_REASONS = ("not_realistic", "pii_remaining", "duplicate_of_existing", "unclear_conversation", "other")


def _submit(store: ReviewStore, record: Record, action: str, reviewer: str, **fields: str | None) -> None:
    seconds = time.time() - st.session_state.get("shown_at", time.time())
    store.record(
        ReviewDecision(
            trace_id=record.trace.trace_id,
            reviewer=reviewer,
            action=action,  # type: ignore[arg-type]
            reject_reason=fields.get("reject_reason"),
            intent=fields.get("intent"),
            failure_mode=fields.get("failure_mode"),
            note=fields.get("note") or "",
            seconds_spent=round(seconds, 1),
            decided_at=datetime.now().isoformat(timespec="seconds"),
            is_rereview=record.meta.rereview,
        )
    )
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Golden-set review", layout="wide")
    run_dir = os.environ.get("GOLDENSET_RUN_DIR")
    if not run_dir:
        st.error("Set GOLDENSET_RUN_DIR to a run folder, or start this page with `goldenset review --run runs/<id>`.")
        return

    store = ReviewStore(Path(run_dir))
    reviewer = st.sidebar.text_input("Reviewer", value="reviewer-1", key="reviewer")
    queue, pending = store.queue(), store.pending()
    done = len(queue) - len(pending)
    st.progress(done / len(queue) if queue else 1.0, text=f"{done} of {len(queue)} reviewed")
    if not pending:
        st.success("Queue complete. Run `goldenset finalize` to build the golden set.")
        return

    record = pending[0]
    item_key = f"{record.trace.trace_id}:{int(record.meta.rereview)}"
    if st.session_state.get("item_key") != item_key:
        st.session_state["item_key"] = item_key
        st.session_state["shown_at"] = time.time()

    enrichment = record.meta.enrichment
    usable = enrichment is not None and not enrichment.failed
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"Trace {record.trace.trace_id}")
        st.text(render_conversation(record.trace))
    with right:
        st.caption(
            f"shape: {record.meta.shape} · represents {record.meta.cluster_size} similar traces · "
            f"redactions: {record.meta.scrub.entity_counts if record.meta.scrub else {}}"
        )
        if enrichment is not None:
            st.caption(f"Model labels ({enrichment.model}), agreement {enrichment.confidence:.1f}")
            if enrichment.ideal_response:
                st.info(f"Ideal response: {enrichment.ideal_response}")
        intent_default = INTENTS.index(enrichment.intent) if enrichment and enrichment.intent in INTENTS else 0
        failure_default = (
            FAILURE_MODES.index(enrichment.failure_mode) if enrichment and enrichment.failure_mode in FAILURE_MODES else 0
        )
        # Keys include the record, so Streamlit never carries one record's selections onto the next.
        intent = st.selectbox("Intent", INTENTS, index=intent_default, key=f"intent-{item_key}")
        failure_mode = st.selectbox("Failure mode", FAILURE_MODES, index=failure_default, key=f"failure_mode-{item_key}")
        reject_reason = st.selectbox("Reject reason", REJECT_REASONS, key=f"reject_reason-{item_key}")
        note = st.text_input("Note", key=f"note-{item_key}")

        accept, correct, reject = st.columns(3)
        if accept.button("Accept", key="accept", type="primary", disabled=not usable):
            _submit(store, record, "accept", reviewer, note=note)
        if correct.button("Save corrections", key="correct"):
            _submit(store, record, "correct", reviewer, intent=intent, failure_mode=failure_mode, note=note)
        if reject.button("Reject", key="reject"):
            _submit(store, record, "reject", reviewer, reject_reason=reject_reason, note=note)


if __name__ == "__main__":  # Streamlit runs the script as __main__; importing the module does nothing
    main()
