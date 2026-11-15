"""Step-0 dynamics brief: one sentence added to the caption, once, before generation.

The half-benchmark showed two things about text: changing it mid-rollout causes
hard resets, and a still-life description freezes the scene. So this touches the
text exactly once, at step 0, and only adds motion: a single sentence, in verbs,
saying what keeps moving in this scene and how the scene physically reacts when
something moves through it (water ripples, grass bends, a queue shuffles forward).
The caption itself is left as it is. This is what lets the method act on cases
with a still camera (insight) or a walking subject (interaction), where rotation
memory has nothing to do.

Agents (shared memory identities, see tools/mem.py):
  cartographer  asks the web (Exa) how the scene's main motion or interaction
                actually looks, and stores the sourced answer
  director      turns that into the one-sentence brief and stores it
  composer      on a later episode of the same scene, recalls the brief instead of
                recomputing it

Inputs are the first frame and the caption only: not the objective, not the action
list, not the rubric.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import exa_search  # noqa: E402
import llm  # noqa: E402
import mem  # noqa: E402

ASK = """You see the first frame of a scene and its caption. A video model will continue this scene. Reply with one JSON object and nothing else:
{"motion": one phrase naming what is already moving or under way in the scene (or "nothing" if it is fully static),
 "question": one short factual question for a web search about how the main motion or physical interaction in this kind of scene actually looks (for example what water does around legs wading into it, how a checkout queue advances, how wet paint behaves), or null if nothing in the scene moves or could be interacted with}"""
WRITE = """Write ONE sentence (20-35 words) to append to a video model's caption so the scene keeps moving and reacts physically. Rules: use active verbs in the present continuous; say what keeps moving and how the surroundings respond when something moves through them or touches them (ripples, splashes, bending, footprints, shadows, items being added or removed); be consistent with the caption and the image; do not introduce new characters or objects; do not mention any camera. Reply with one JSON object and nothing else: {"brief": "..."}"""


def make(case, image, use_web=True, use_memory=True, log=None):
    """Returns {"brief": sentence or "", "source": ..., "url": ..., "question": ...}."""
    key = case["task_id"]
    if use_memory:
        try:
            for h in mem.recall("composer", f"{key} dynamics brief", 6):
                for text in [h.get("text") or ""] + list(h.get("excerpts") or []):
                    m = re.search(rf"{key} dynamics brief: (.+)", text)
                    if m:
                        return {"brief": m.group(1).strip(), "source": f"past.dev:{h.get('source_id')}",
                                "url": None, "question": None}
        except Exception:
            pass
    a = llm.ask_json(ASK, f"Caption: {case['image_caption']}", [image], max_tokens=300)
    question, answer, url = a.get("question"), None, None
    if question and use_web:
        try:
            res = exa_search.answer(question)
            answer = res.get("answer")
            url = (res.get("citations") or [{}])[0].get("url")
            if use_memory:
                mem.write("cartographer", "fact", f"{key} physical behaviour: {question} -> {str(answer)[:400]}",
                          {"case": key, "url": url})
        except Exception:
            answer = None
    if a.get("motion") in (None, "nothing") and not answer:
        return {"brief": "", "source": "none", "url": None, "question": question}
    w = llm.ask_json(WRITE, f"Caption: {case['image_caption']}\nAlready moving: {a.get('motion')}\n"
                     f"Sourced fact about how it behaves: {answer or '(none)'}", [image], max_tokens=200)
    brief = (w.get("brief") or "").strip()
    if brief and use_memory:
        try:
            mem.write("director", "state", f"{key} dynamics brief: {brief}", {"case": key, "url": url})
        except Exception:
            pass
    return {"brief": brief, "source": "exa+llm" if answer else "llm", "url": url, "question": question}
