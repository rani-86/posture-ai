"""PostureAI Streamlit dashboard.

Calls src/inference directly rather than going through the FastAPI backend (Phase 7).
Decision, documented here rather than left implicit: a Streamlit app is already a single
Python process, so adding an HTTP round-trip to its own sibling API would only add
latency and a second process to keep running, for no real benefit in this project's
scope. The FastAPI backend (Phase 7) remains the integration surface for a separate
client/service; this dashboard is a second, independent consumer of the same
src/inference code, not a wrapper around the API.

Run with:
    .venv/Scripts/python.exe -m streamlit run app/dashboard.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference.report_pipeline import VideoAnalysisReport, analyze_video
from src.utils.config import load_config

st.set_page_config(page_title="PostureAI", layout="wide")


@st.cache_resource
def _config():
    return load_config()


def _frame_records_df(report: VideoAnalysisReport) -> pd.DataFrame:
    return pd.DataFrame([vars(r) for r in report.frame_records])


def render_overview() -> None:
    st.title("PostureAI")
    st.markdown(
        """
PostureAI analyzes an uploaded video frame-by-frame: MediaPipe pose estimation ->
joint-angle and posture-geometry feature extraction -> a rule-based posture-deviation
engine -> an activity-recognition model, with an annotated skeleton-overlay video and a
frame-level timeline.

**What's real vs. placeholder in this build** (full detail in `docs/phase1_notes.md`
through `docs/phase7_notes.md`):
- Pose estimation, joint angles, and rule-based posture analysis: run on **your actual
  uploaded video**, validated against synthetic-geometry unit tests (no clinical
  threshold validation).
- Movement-type classifier (XGBoost/LSTM/GRU, 93-94% accuracy): trained on **real**
  UI-PRMD data, but in a feature space (Kinect Euler angles) that doesn't match this
  app's own video pipeline - **not used for live predictions here**, see Phase 7.
- Activity classifier (sitting/standing/walking/bending/squatting) shown below: trained
  **only on synthetic placeholder data** - real labeled clips were never available for
  this project. Its prediction is a pipeline demonstration, not a validated result.
"""
    )


def render_upload_and_preview() -> Path | None:
    st.header("1. Upload a video")
    uploaded = st.file_uploader("MP4 video", type=["mp4"])
    if uploaded is None:
        st.info("Upload an .mp4 file to run the analysis.")
        return None

    tmp_dir = Path(tempfile.mkdtemp())
    video_path = tmp_dir / "input.mp4"
    video_path.write_bytes(uploaded.getvalue())

    st.header("2. Preview")
    st.video(uploaded)
    return video_path


def render_pose_and_posture(report: VideoAnalysisReport) -> None:
    st.header("3. Pose estimation")
    if report.annotated_video_path and Path(report.annotated_video_path).exists():
        st.video(report.annotated_video_path)
    detection_rate = report.posture_summary.frames_with_pose_detected / report.posture_summary.total_frames
    st.caption(f"Pose detected in {detection_rate:.0%} of {report.posture_summary.total_frames} frames.")
    for w in report.warnings:
        st.warning(w)

    st.header("4. Posture analysis")
    st.caption(
        "Rule-based, configurable-threshold deviations (see docs/phase3_notes.md for camera-angle "
        "assumptions per feature - this is not a clinical diagnosis)."
    )
    cols = st.columns(4)
    summary = report.posture_summary
    cols[0].metric("Forward-head deviation", _pct(summary.forward_head_deviation_rate))
    cols[1].metric("Shoulder asymmetry", _pct(summary.shoulder_asymmetry_deviation_rate))
    cols[2].metric("Trunk inclination deviation", _pct(summary.trunk_inclination_deviation_rate))
    cols[3].metric(
        "Mean trunk inclination",
        f"{summary.mean_trunk_inclination_deg:.1f}°" if summary.mean_trunk_inclination_deg is not None else "N/A",
    )


def _pct(value: float | None) -> str:
    return f"{value:.0%} of frames" if value is not None else "N/A"


def render_joint_angles_and_timeline(report: VideoAnalysisReport) -> None:
    df = _frame_records_df(report)
    if df.empty:
        st.info("No frame data to chart.")
        return

    st.header("5. Joint-angle analytics")
    angle_cols = ["left_knee_deg", "right_knee_deg", "trunk_inclination_deg", "forward_head_angle_deg"]
    available = [c for c in angle_cols if df[c].notna().any()]
    if available:
        st.line_chart(df.set_index("timestamp_sec")[available])
    else:
        st.info("No joint-angle data available (no pose detected in any frame).")

    st.header("6. Temporal movement timeline")
    timeline_df = df[["timestamp_sec", "pose_detected", "any_deviation"]].copy()
    timeline_df["pose_detected"] = timeline_df["pose_detected"].astype(int)
    timeline_df["any_deviation"] = timeline_df["any_deviation"].fillna(0).astype(int)
    st.area_chart(timeline_df.set_index("timestamp_sec"))
    st.caption("1 = detected / deviation flagged that frame, 0 = not detected / no deviation / undetermined.")


def render_activity_recognition(report: VideoAnalysisReport) -> None:
    st.header("7. Activity recognition")
    pred = report.activity_prediction
    if pred is None:
        st.info("Activity model unavailable for this run.")
        return
    st.warning(pred.disclaimer)
    st.metric("Predicted activity", pred.predicted_label, f"{pred.confidence:.0%} confidence")
    probs_df = pd.DataFrame({"activity": list(pred.class_probabilities.keys()), "probability": list(pred.class_probabilities.values())})
    st.bar_chart(probs_df.set_index("activity"))


def render_detailed_report(report: VideoAnalysisReport) -> None:
    st.header("8. Detailed report")
    with st.expander("Raw report data"):
        st.json(
            {
                "posture_summary": vars(report.posture_summary),
                "activity_prediction": vars(report.activity_prediction) if report.activity_prediction else None,
                "warnings": report.warnings,
                "num_frames_recorded": len(report.frame_records),
            }
        )
    df = _frame_records_df(report)
    if not df.empty:
        st.download_button(
            "Download frame-level data (CSV)", df.to_csv(index=False), file_name="posture_ai_frame_report.csv"
        )


def main() -> None:
    render_overview()
    video_path = render_upload_and_preview()
    if video_path is None:
        return

    if st.button("Run analysis", type="primary"):
        config = _config()
        annotated_path = video_path.parent / "annotated.mp4"
        with st.spinner("Processing video (pose estimation, posture analysis, activity prediction)..."):
            try:
                report = analyze_video(video_path, config, annotated_output_path=annotated_path)
            except ValueError as e:
                st.error(str(e))
                return
        st.session_state["report"] = report

    report = st.session_state.get("report")
    if report is not None:
        render_pose_and_posture(report)
        render_joint_angles_and_timeline(report)
        render_activity_recognition(report)
        render_detailed_report(report)


if __name__ == "__main__":
    main()
