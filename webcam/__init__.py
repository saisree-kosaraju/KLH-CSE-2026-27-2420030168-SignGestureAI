"""
webcam/ — offline diagnostics, preprocessing and benchmark tooling for making the
frozen INCLUDE262 model work on real webcam input.

Nothing in this package modifies datasets/, metadata/, processed_data/INCLUDE262/,
models/ or scripts/. It reuses the existing pipeline (scripts 76 + 79, driven through
app/backend/) and only writes new webcam_* artifacts under results/, processed_data/WEBCAM/
and webcam_data/.
"""
