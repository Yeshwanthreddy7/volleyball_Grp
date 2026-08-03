"""
detect_utils.py - torch-free detection helpers.

Kept separate from pipeline.py so the class-id resolution logic can be unit
tested without importing torch / ultralytics / cv2.
"""
from __future__ import annotations

PERSON_ALIASES = {"person", "player", "players", "people", "human", "athlete"}


def resolve_class_ids(names, person_override=None, ball_override=None):
    """Resolve (person_id, ball_id) for ANY detector from its class names.

    Prevents the custom-model loophole: a Roboflow/YOLOv11 model does NOT use
    COCO ids. Roboflow orders classes alphabetically, so a 2-class set is
    typically {0: 'ball', 1: 'player'} - the opposite of COCO (person=0,
    ball=32). Looking ids up BY NAME makes the pipeline correct for COCO and for
    any custom model; explicit overrides always win.

    Parameters
    ----------
    names : dict {class_id: class_name}  (ultralytics `model.names`)
    person_override, ball_override : optional explicit ids (take priority)

    Returns
    -------
    (person_id, ball_id) - ball_id may be None if the model has no ball class.
    """
    person_id = person_override
    ball_id = ball_override
    if person_id is None:
        for i, n in names.items():
            if str(n).strip().lower() in PERSON_ALIASES:
                person_id = int(i)
                break
    if ball_id is None:
        for i, n in names.items():
            if "ball" in str(n).strip().lower():   # ball, volleyball, sports ball
                ball_id = int(i)
                break
    if person_id is None and len(names) == 1:       # single-class detector
        person_id = int(next(iter(names)))
    return person_id, ball_id
