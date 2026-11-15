"""Object ledger: what is in the scene, where, and which of it is supposed to change.

Built once at step 0 from the first frame and the caption (never the objective, the action list or the
rubric). It drives two things in the episode:

  * the write-back on return is by region: the parts of the frame the ledger calls static are pinned
    from the stored view (object permanence), while the region of anything moving or mid-process is
    left for the model to render, so it is not frozen at its old state;
  * what the moving thing should look like by then comes from Exa (how fast that process advances) and
    enters as the return register.

Agents (shared memory identities, tools/mem.py):
  cartographer  lists the objects and their regions; asks Exa about the one in progress; stores both
  director      writes the sentence for the state on return
  critic        after the rollout, checks each ledger object on the returned view and stores what it saw
  controller    reads the critic's record for this scene before a second attempt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))

ASK = """You see the first frame of a scene and its caption. List up to 6 distinct things a viewer would expect to
still find when looking back at this view later: the main subject and the most recognisable objects or structures.
Reply with one JSON object and nothing else:
{"objects": [{"name": short name with its distinguishing colour or feature,
              "box": [x0, y0, x1, y1] as fractions of the image width and height (0 to 1), a tight box around it,
              "kind": "static" if it should look the same later (furniture, buildings, sculptures, parked things),
                      "moving" if it is travelling through the scene (a vehicle under way, a walking person, an animal),
                      "process" if it is in the middle of a task whose result should visibly accumulate
                               (drawing, pouring, cutting, cooking, filling, building)}]}"""
CHECK = """You are checking whether a video kept its scene. Image 1 is the first frame. Image 2 shows frames from the end of
the video, after the camera looked away and came back. For each listed object say what happened to it.
Objects: {objects}
Reply with one JSON object and nothing else:
{{"objects": [{{"name": as given, "status": "same" | "changed_plausibly" | "frozen" | "replaced" | "missing"}}],
  "returned": true if image 2 shows the same place as image 1 from about the same viewpoint, else false}}
Use "changed_plausibly" when a moving or in-progress thing has visibly moved on or progressed, "frozen" when such a
thing looks exactly as in image 1, "replaced" when something different stands where it was, "missing" when it is gone."""


def _box(b):
    try:
        x0, y0, x1, y1 = [min(1.0, max(0.0, float(v))) for v in b]
    except Exception:
        return None
    if x1 - x0 < 0.02 or y1 - y0 < 0.02:
        return None
    return [x0, y0, x1, y1]


def make(case, image, use_web=True, use_memory=True):
    """Returns {"objects": [...], "sentence": str, "entity": ..., "url": ..., "source": ...}."""
    import llm
    import outsight
    out = {"objects": [], "sentence": "", "entity": None, "url": None, "source": "none"}
    try:
        v = llm.ask_json(ASK, f"Caption: {case['image_caption']}", [image], max_tokens=700)
        for o in (v.get("objects") or [])[:6]:
            b = _box(o.get("box"))
            if b and o.get("kind") in ("static", "moving", "process"):
                out["objects"].append({"name": str(o.get("name"))[:80], "box": b, "kind": o["kind"]})
    except Exception as e:
        out["source"] = "ledger failed: " + repr(e)[:100]
    if any(o["kind"] != "static" for o in out["objects"]):
        try:
            s = outsight.make(case, image, use_web=use_web, use_memory=use_memory)
            out.update(sentence=s.get("text") or "", entity=s.get("entity"), url=s.get("url"), source=s.get("source"),
                       question=s.get("question"), fact=s.get("fact"))
        except Exception as e:
            out["source"] = "sentence failed: " + repr(e)[:100]
    if use_memory and out["objects"]:
        try:
            import mem
            mem.write("cartographer", "fact", f"{case['task_id']} object ledger: " + "; ".join(
                f"{o['name']} ({o['kind']}) at {[round(x, 2) for x in o['box']]}" for o in out["objects"]),
                meta={"case": case["task_id"], "objects": len(out["objects"])})
        except Exception:
            pass
    return out


def pin_mask(objects, h, w, grow=0.06):
    """[h, w] float mask on the latent grid: 1 where the stored view may be pinned, 0 over anything the
    ledger calls moving or in process (its box grown by `grow` of the frame on each side)."""
    m = np.ones((h, w), np.float32)
    for o in objects:
        if o["kind"] == "static":
            continue
        x0, y0, x1, y1 = o["box"]
        a, b = int(max(0, (y0 - grow)) * h), int(np.ceil(min(1, y1 + grow) * h))
        c, d = int(max(0, (x0 - grow)) * w), int(np.ceil(min(1, x1 + grow) * w))
        m[a:b, c:d] = 0.0
    return m


def check(case, first_image, frames, objects, use_memory=True):
    """Critic: per-object status on the returned view. Returns {"objects": [...], "returned": bool, "score": 0..1}."""
    import llm
    from PIL import Image
    if not objects:
        return {"objects": [], "returned": None, "score": None}
    tail = [np.asarray(frames[i]) for i in np.linspace(max(0, len(frames) - 24), len(frames) - 1, 4).astype(int)]
    hh, ww = tail[0].shape[:2]
    sheet = Image.new("RGB", (ww * 2, hh * 2))
    for j, f in enumerate(tail):
        sheet.paste(Image.fromarray(f), ((j % 2) * ww, (j // 2) * hh))
    v = llm.ask_json("You compare images literally and reply with JSON only.",
                     CHECK.format(objects="; ".join(f"{o['name']} ({o['kind']})" for o in objects)),
                     [first_image, np.asarray(sheet)], max_tokens=500)
    res = []
    for o in objects:
        st = next((x.get("status") for x in (v.get("objects") or []) if str(x.get("name"))[:20] == o["name"][:20]), None)
        res.append({"name": o["name"], "kind": o["kind"], "status": st})
    good = sum(1 for r in res if (r["kind"] == "static" and r["status"] == "same") or
               (r["kind"] != "static" and r["status"] == "changed_plausibly"))
    out = {"objects": res, "returned": bool(v.get("returned")), "score": round(good / max(1, len(res)), 2)}
    if use_memory:
        try:
            import mem
            bad = [f"{r['name']}: {r['status']}" for r in res if r["status"] not in ("same", "changed_plausibly")]
            mem.write("critic", "defect", f"{case['task_id']} on return: " + ("; ".join(bad) if bad else "every ledger object accounted for")
                      + f". Returned to the same view: {out['returned']}.", meta={"case": case["task_id"], "score": out["score"]})
        except Exception:
            pass
    return out
