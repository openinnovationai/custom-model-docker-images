from typing import List

from pydantic import BaseModel


class RTTMSegment(BaseModel):
    """Structured representation of a single RTTM diarization line.

    Fields follow the standard RTTM columns for SPEAKER entries.
    """
    type: str  # e.g., "SPEAKER"
    file_id: str
    channel_id: int
    turn_onset: float
    turn_duration: float
    orthography_field: str
    speaker_type: str
    speaker_name: str
    confidence_score: str
    signal_lookahead_time: str


def convert_annotation_to_segments(annotation, file_id: str) -> List[RTTMSegment]:
    """Convert pyannote Annotation directly into RTTMSegment list.

    This avoids serializing to RTTM text first and uses the structured
    segments produced by pyannote directly.
    """
    segments: List[RTTMSegment] = []
    # Prefer annotation-provided URI if available
    try:
        annotation_uri = getattr(annotation, "uri", None)
        if isinstance(annotation_uri, str) and annotation_uri:
            file_id = annotation_uri
    except Exception:
        pass
    # Optional modality info (not standard RTTM field but can hint speaker_type)
    modality_hint = None
    try:
        modality_hint = getattr(annotation, "modality", None)
    except Exception:
        modality_hint = None

    # pyannote Annotation supports itertracks(yield_label=True)
    # which yields (segment, track, label)
    try:
        for segment, track, label in annotation.itertracks(yield_label=True):
            onset = float(segment.start)
            duration = float(segment.end - segment.start)
            # Try to obtain confidence score from attached scores structure if present
            confidence_value = "<NA>"
            try:
                scores_obj = getattr(annotation, "scores_", None) or getattr(
                    annotation, "scores", None) or getattr(annotation, "_scores", None)
                if scores_obj is not None:
                    get_fn = getattr(scores_obj, "get", None)
                    if callable(get_fn):
                        val = get_fn(segment, track, label)
                        if isinstance(val, (int, float)):
                            confidence_value = f"{float(val):.6f}"
            except Exception:
                confidence_value = "<NA>"

            segments.append(
                RTTMSegment(
                    type="SPEAKER",
                    file_id=file_id,
                    channel_id=0,
                    turn_onset=onset,
                    turn_duration=duration,
                    orthography_field=str(
                        label) if label is not None else "<NA>",
                    speaker_type=str(
                        modality_hint) if modality_hint else "speaker",
                    speaker_name=str(label) if label is not None else "<NA>",
                    confidence_score=confidence_value,
                    signal_lookahead_time="0",
                )
            )
    except Exception:
        # Fallback: try itersegments + labels
        try:
            for segment in annotation.itersegments():
                onset = float(segment.start)
                duration = float(segment.end - segment.start)
                segments.append(
                    RTTMSegment(
                        type="SPEAKER",
                        file_id=file_id,
                        channel_id=0,
                        turn_onset=onset,
                        turn_duration=duration,
                        orthography_field="<NA>",
                        speaker_type="<NA>",
                        speaker_name="<NA>",
                        confidence_score="<NA>",
                        signal_lookahead_time="<NA>",
                    )
                )
        except Exception:
            return []
    return segments
