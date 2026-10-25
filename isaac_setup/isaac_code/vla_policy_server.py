#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.util
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import load_config, pack, resolve, unpack


def load_adapter(cfg, base):
    pcfg = cfg["vla_policy"]
    path = resolve(pcfg["adapter"], base)
    spec = importlib.util.spec_from_file_location("tomo_vla_adapter", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    policy = mod.load_policy(
        pcfg.get("device", "cuda:0"),
        str(resolve(pcfg["checkpoint"], base)),
        pcfg.get("text_model", "google/siglip-base-patch16-224"),
        pcfg.get("text_device", "cpu"),
        bool(pcfg.get("local_files_only", True)),
        str(resolve(pcfg["language_bank"], base)) if pcfg.get("language_bank") else None,
    )
    return mod, policy


def serve(cfg, base):
    pcfg = cfg["vla_policy"]
    mod, policy = load_adapter(cfg, base)
    addr = (pcfg.get("host", "127.0.0.1"), int(pcfg.get("port", 6031)))
    listener = Listener(addr, authkey=str(pcfg.get("authkey", "tomo-vla")).encode())
    print(f"[vla] READY on {addr[0]}:{addr[1]}", flush=True)

    try:
        sys.path.append("/home/emagesoft/Downloads/flow_matching_vla")
        from llm_model import VLASemanticRouter
        semantic_router = VLASemanticRouter()
    except Exception as e:
        semantic_router = None
        print(f"Warning: Semantic router failed to load: {e}")

    while True:
        conn = listener.accept()
        history = []
        try:
            while True:
                try:
                    msg = conn.recv()
                except EOFError:
                    break
                cmd = msg.get("cmd")
                if cmd == "hello":
                    conn.send({
                        "ok": True,
                        "joint_names": list(mod.JOINT_NAMES),
                        "checkpoint": str(pcfg["checkpoint"]),
                        "horizon": int(mod.HORIZON),
                        "suction_trained": True,
                        "done_trained": True,
                    })
                elif cmd == "get_user_input":
                    if semantic_router:
                        print("\n" + "="*50)
                        print("🤖 VLA Interactive Controller (Terminal 1)")
                        print("Type your command for the robot below (or 'exit' to quit)")
                        print("="*50)
                        user_input = input("\nYou: ")
                        if user_input.strip().lower() in ['exit', 'quit']:
                            conn.send({"command": "exit"})
                        else:
                            result = semantic_router.parse_user_command(user_input)
                            if result:
                                print(f"-> Executing: {result['matched_intent']} | Stop Condition: {result['stop_condition']}")
                                conn.send({"command": "run", "vla_prompt": result["vla_prompt"], "stop_condition": result["stop_condition"]})
                            else:
                                print("-> Unknown command. Executing default.")
                                conn.send({"command": "run", "vla_prompt": pcfg.get("default_instruction", ""), "stop_condition": "done"})
                    else:
                        user_input = input("\n[Terminal 1] Enter command (or 'exit'): ")
                        if user_input.strip().lower() in ['exit', 'quit']:
                            conn.send({"command": "exit"})
                        else:
                            conn.send({"command": "run", "vla_prompt": user_input, "max_frames": 307})
                elif cmd == "reset":
                    history.clear()
                    conn.send({"ok": True})
                elif cmd in ("observe", "infer"):
                    t0 = time.perf_counter()
                    if "rgb" in msg:
                        obs = mod.build_obs(
                            unpack(msg["rgb"]),
                            unpack(msg["depth"]),
                            int(msg["frame_id"]),
                        )
                        q = np.asarray(msg["q"], np.float32)
                        history.append((int(msg["frame_id"]), obs, q))
                        del history[:-int(mod.N_OBS)]
                    reply = {"ok": True}
                    if cmd == "infer":
                        hist = [history[0]] * (int(mod.N_OBS) - len(history)) + history
                        out = policy.predict(
                            [x[1] for x in hist],
                            np.stack([x[2] for x in hist]),
                            msg.get("instruction", pcfg.get("default_instruction", "")),
                            int(pcfg.get("inference_seed", 12345)) + hist[-1][0],
                        )
                        reply.update({
                            "horizon": pack(out["horizon"]),
                            "suction": pack(out["suction"]),
                            "done": pack(out["done"]),
                            "phone_pose": out["phone_pose"].tolist(),
                            "phone_patch": out["phone_patch"],
                            "infer_ms": round((time.perf_counter() - t0) * 1e3, 2),
                        })
                    conn.send(reply)
                elif cmd == "shutdown":
                    conn.send({"ok": True})
                    conn.close()
                    listener.close()
                    return
                else:
                    conn.send({"ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            try:
                conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config_vla.yaml"))
    args = ap.parse_args()
    cfg, base = load_config(args.config)
    serve(cfg, base)


if __name__ == "__main__":
    main()
