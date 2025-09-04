#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=${PYANNOTE_MODEL_ID:-pyannote/speaker-diarization-3.1}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required at build-time to pre-download ${MODEL_ID}" >&2
  exit 1
fi

python3 - <<'PY'
import os
from pyannote.audio import Pipeline
model_id = os.environ.get('PYANNOTE_MODEL_ID', 'pyannote/speaker-diarization-3.1')
token = os.environ.get('HF_TOKEN')
Pipeline.from_pretrained(model_id, use_auth_token=token)
print('Pre-downloaded', model_id)
PY


