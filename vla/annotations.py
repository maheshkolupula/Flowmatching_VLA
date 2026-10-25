"""Skill segments, suction labels and phone labels for the TOMO VLA.

The Find_and_Pick demonstrations share ONE physical timeline (verified on all 46 raw episodes:
T=307 at 30 fps, io_events [[155,128,200],[290,0,200]], all joints bit-identical across episodes
except the right arm over frames 116-200, and the phone start pose):

    head moves 11-40 and 51-79, both arms to the ready pose 25-45
    right arm rises 91-110 and moves over the phone 116-132, hovers 133-137, descends 138-154
    suction ON 155, dwell 155-160, phone moves from 161, rises 161-195, hold pose reached at 201
    hold 202-261 (left arm 228-247), transport 262-282, lowering 282-288
    suction OFF 290, phone drops 291-293, retract 294-306

The IK moved-phone demos keep this timeline (only the right arm is re-solved over 60-245), so the
same boundaries apply to them. Boundaries are therefore fixed frame numbers (TIMELINE) anchored
on the recorded suction events, instead of a motion detector (the bundle's detector put
reach_start at 137 and ended CARRY at peak hand speed).
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import yaml

JOINT_PREFIXES = ("Head_joint", "Left_arm_joint", "Right_arm_joint", "RF_joint_", "LF_joint_")
PHONE_KEYS = ["iPhone.location.x", "iPhone.location.y", "iPhone.location.z",
              "iPhone.rotation.x", "iPhone.rotation.y", "iPhone.rotation.z"]

# Frame boundaries (inclusive). suction_on / suction_off come from io_events when present.
TIMELINE = {
    "reach_start": 91,      # right arm starts rising after the ready pause
    "suction_on": 155,
    "phone_moves": 161,     # first phone motion after the vacuum dwell
    "hold_reached": 201,    # phone lifted to the hold pose
    "place_start": 282,     # transport over the box finished, lowering starts
    "suction_off": 290,
}

TRAIN_INSTRUCTIONS = {
    "LOOK_READY": ["look at the work area", "look down at the table", "get ready to pick up the phone",
                   "prepare to manipulate the phone"],
    "REACH_PHONE": ["reach the phone", "move toward the phone", "move the hand to the phone",
                    "approach the iPhone", "go to the phone"],
    "GRASP_PHONE": ["grasp the phone with suction", "attach the suction cup to the phone",
                    "turn on the suction", "hold the phone with the suction cup"],
    "PICK_UP_PHONE": ["pick up the phone", "pick up the iPhone", "grab the phone from the table",
                      "take the phone"],
    "LIFT_PHONE": ["lift the phone", "raise the iPhone", "move the phone upward", "lift the phone up"],
    "CARRY_TO_BOX": ["carry the phone to the box", "move the phone toward the box",
                     "transport the iPhone to the carton", "bring the phone over the box"],
    "PLACE_IN_BOX": ["put the phone in the box", "place the iPhone inside the box",
                     "lower the phone into the carton", "drop the phone into the box"],
    "RETRACT": ["retract the arm", "move the arm away", "return after placing the phone"],
    "FULL_TASK": ["pick up the phone and put it in the box", "put the iPhone in the box",
                  "pack the phone", "place the phone into the carton",
                  "move the phone from the table into the box"],
}

# Held-out paraphrases: NEVER trained, only used by the language test to check generalisation.
TEST_INSTRUCTIONS = {
    "REACH_PHONE": ["move your hand over to the phone"],
    "PICK_UP_PHONE": ["pick the phone up off the table"],
    "LIFT_PHONE": ["raise the phone into the air"],
    "PLACE_IN_BOX": ["set the phone down inside the box"],
    "FULL_TASK": ["put the phone that is on the table into the box"],
}


def parse_mmrt_full(path: Path) -> dict:
    """46 joints (yaml order == cache order, verified), 6-D phone pose per frame, suction events."""
    doc = yaml.safe_load(Path(path).read_text())
    labels = doc.get("labels", {})
    robot_labels = [str(x) for x in labels.get("robots", [])]
    active_labels = [str(x) for x in labels.get("active_objects", [])]
    frames = np.asarray(doc.get("frames", []), dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != len(robot_labels) + len(active_labels):
        raise ValueError(f"{path}: frame width {frames.shape} != robots {len(robot_labels)} + objects {len(active_labels)}")
    joint_names = [n for n in robot_labels if n.startswith(JOINT_PREFIXES)]
    if len(joint_names) != 46:
        raise ValueError(f"{path}: expected 46 joints, got {len(joint_names)}")
    q = frames[:, [robot_labels.index(n) for n in joint_names]]
    cols = {n: len(robot_labels) + i for i, n in enumerate(active_labels)}
    if not all(k in cols for k in PHONE_KEYS):
        raise ValueError(f"{path}: iPhone pose labels missing")
    phone6 = frames[:, [cols[k] for k in PHONE_KEYS]]
    on = off = None
    for e in doc.get("io_events", []) or []:
        frame, value = int(e[0]), float(e[1])
        if on is None and value != 0:
            on = frame
        elif on is not None and off is None and value == 0 and frame > on:
            off = frame
    return {"joint_names": joint_names, "q": q.astype(np.float32), "phone_pose6": phone6.astype(np.float32),
            "suction_on": on, "suction_off": off, "n_frames": int(len(frames))}


def boundaries_for(n_frames: int, suction_on=None, suction_off=None) -> dict:
    b = dict(TIMELINE)
    if suction_on is not None:
        b["suction_on"] = int(suction_on)
    if suction_off is not None:
        b["suction_off"] = int(suction_off)
    b["start"], b["end"] = 0, int(n_frames) - 1
    order = ["start", "reach_start", "suction_on", "phone_moves", "hold_reached", "place_start", "suction_off", "end"]
    vals = [b[k] for k in order]
    if vals != sorted(vals) or len(set(vals)) != len(vals):
        raise ValueError(f"timeline boundaries out of order for T={n_frames}: {b}")
    return b


def make_segments(b: dict) -> list[dict]:
    rs, on, mv, hold, ps, off, end = (b[k] for k in ("reach_start", "suction_on", "phone_moves", "hold_reached",
                                                     "place_start", "suction_off", "end"))
    raw = [
        ("LOOK_READY", 0, rs - 1),        # look down + both arms to the ready pose, then wait
        ("REACH_PHONE", rs, on - 1),      # rise, move over the phone, hover, descend (phone dependent)
        ("GRASP_PHONE", on, mv - 1),      # suction on + dwell, phone still on the table
        ("PICK_UP_PHONE", rs, hold),      # reach + attach + lift to the hold pose
        ("LIFT_PHONE", on, hold),         # attach + lift
        ("CARRY_TO_BOX", hold + 1, ps),   # hold, then transport over the box (ends at rest, above the box)
        ("PLACE_IN_BOX", ps, off),        # lower + release (ends at suction OFF)
        ("RETRACT", off, end),
        ("FULL_TASK", 0, end),
    ]
    out = []
    for name, s, e in raw:
        if e <= s:
            raise ValueError(f"degenerate segment {name} {s}-{e}")
        out.append({"name": name, "start": int(s), "end": int(e),
                    "instructions": list(TRAIN_INSTRUCTIONS[name]),
                    "test_instructions": list(TEST_INSTRUCTIONS.get(name, []))})
    return out


def suction_from_events(n_frames: int, on: int, off: int) -> np.ndarray:
    s = np.zeros(n_frames, dtype=np.float32)
    s[on:off] = 1.0
    return s


def phone_pose_target(phone6: np.ndarray) -> np.ndarray:
    """[T,6] (x, y, z, rx, ry, rz) -> [T,4] (x, y, sin yaw, cos yaw); yaw spans 0.8-2.2 rad, no wrap."""
    p = np.asarray(phone6, dtype=np.float32)
    return np.stack([p[:, 0], p[:, 1], np.sin(p[:, 5]), np.cos(p[:, 5])], 1).astype(np.float32)


def write_episode_labels(ep_dir: Path, n_frames: int, suction_on: int, suction_off: int,
                         phone6: np.ndarray, phone_uv: np.ndarray | None = None,
                         phone_visible: np.ndarray | None = None) -> dict:
    """vla_labels.npz + segments.json for one episode folder.
    phone_uv: [T,2] pixel centre in the native 960x720 image (-1 when unknown)
    phone_visible: [T] 1 visible / 0 not visible (outside the image or hidden) / -1 unknown"""
    b = boundaries_for(n_frames, suction_on, suction_off)
    segs = make_segments(b)
    if phone_uv is None:
        phone_uv = np.full((n_frames, 2), -1.0, np.float32)
    if phone_visible is None:
        phone_visible = np.full(n_frames, -1, np.int8)
    if len(phone6) != n_frames or len(phone_uv) != n_frames or len(phone_visible) != n_frames:
        raise ValueError(f"{ep_dir}: label lengths differ from T={n_frames}")
    np.savez_compressed(Path(ep_dir) / "vla_labels.npz",
                        suction=suction_from_events(n_frames, b["suction_on"], b["suction_off"]),
                        phone_pose=phone_pose_target(phone6),
                        phone_pose6=np.asarray(phone6, np.float32),
                        phone_uv=np.asarray(phone_uv, np.float32),
                        phone_visible=np.asarray(phone_visible, np.int8))
    (Path(ep_dir) / "segments.json").write_text(json.dumps({"boundaries": b, "segments": segs}, indent=2))
    return b


def all_instruction_texts(include_test: bool = True) -> list[str]:
    texts = {t for v in TRAIN_INSTRUCTIONS.values() for t in v}
    if include_test:
        texts |= {t for v in TEST_INSTRUCTIONS.values() for t in v}
    return sorted(texts)
