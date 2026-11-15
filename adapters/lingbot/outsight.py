"""Side memory, text channel: what should the scene look like when the camera comes back?

The model keeps three seconds of rolling attention. Once the camera has looked away for
longer, the only trace of the starting view is the first frames, so on return the model
either copies them (nothing has progressed) or invents (things get replaced). This module
supplies the missing piece of state as text, to be written into the return register (see
episode.py, `ret`): one sentence saying how the *same* things look a few seconds later.

Agents (shared memory identities, tools/mem.py):
  cartographer  names the thing in the first frame that is in the middle of a process,
                asks the web (Exa) how that process advances over a few seconds, and
                stores the sourced answer
  director      writes the one-sentence later state from it and stores it
  composer      on a later episode of the same scene, recalls the sentence

Inputs are the first frame and the caption only: not the objective, not the action list,
not the rubric.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import exa_search  # noqa: E402
import llm  # noqa: E402
import mem  # noqa: E402

ASK = """You see the first frame of a scene and its caption. Reply with one JSON object and nothing else:
{"entity": the one thing in the scene that is moving or in the middle of a process (a vehicle under way, a person doing a task, an animal, a launch, something being made or consumed), named with its distinguishing colours, or null if the scene is fully static,
 "now": a short phrase for its state in this frame,
 "question": one short factual question for a web search about how far that motion or process advances in about ten seconds (speed, rate, what visibly changes), or null}"""
WRITE = """A camera looks away from this scene for a few seconds and then looks back. Write ONE sentence (18-32 words) describing what the camera should see when it looks back: the same entity, the same individual with the same colours and clothing, in the state it has plausibly reached about eight seconds later. Rules: present tense; name the entity and its distinguishing features so it stays the same one; state the visible progress concretely (moved further along, more of the task done, higher, closer, further); keep everything else in the scene as the caption describes; do not introduce new objects or characters; do not mention any camera. Reply with one JSON object and nothing else: {"later": "..."}"""


ASK_ACT = """You see the first frame of a scene and its caption. A video model will continue this scene for 30 seconds
with a still camera. Reply with one JSON object and nothing else:
{"doing": one phrase naming what the visible people, animals or machines are in the middle of doing (or "nothing"
          if the scene is empty of activity),
 "question": one short factual question for a web search about how that activity visibly proceeds step by step
             over about half a minute (what happens next, what accumulates, what gets used up), or null}"""
WRITE_ACT = """Write ONE sentence (22-36 words) telling a video model what the subjects in this scene do next, as an action
in progress. Rules: present continuous, active verbs; name the subject by its distinguishing features and say
what it does step by step; state the visible result building up as it goes (more is eaten, poured, written,
packed, assembled; the queue moves up; the vehicle travels on); use only people and things already in the frame;
do not describe a finished state; do not mention any camera. Reply with one JSON object and nothing else:
{"action": "..."}"""


def action(case, image, use_web=True, use_memory=True):
    """The request for a still-camera scene: the benchmark says only "wait and observe", so what the subjects
    should be doing comes from the scene itself and from Exa (how the activity actually proceeds). Returned as
    one action sentence for the cross-attention register, the same port that makes third-person subjects carry
    out a stated action. {"text", "doing", "question", "fact", "url", "source"}."""
    key = case["task_id"]
    if use_memory:
        try:
            for h in mem.recall("composer", f"{key} action in progress", 6):
                for text in [h.get("text") or ""] + list(h.get("excerpts") or []):
                    m = re.search(rf"{key} action in progress: (.+)", text)
                    if m:
                        return {"text": m.group(1).strip(), "doing": None, "source": f"past.dev:{h.get('source_id')}",
                                "url": None, "question": None, "fact": None}
        except Exception:
            pass
    a = llm.ask_json(ASK_ACT, f"Caption: {case['image_caption']}", [image], max_tokens=300)
    doing, question, answer, url = a.get("doing"), a.get("question"), None, None
    if not doing or str(doing).lower() in ("nothing", "null", "none"):
        return {"text": "", "doing": None, "source": "no activity", "url": None, "question": None, "fact": None}
    if question and use_web:
        try:
            res = exa_search.answer(question)
            answer = res.get("answer")
            url = (res.get("citations") or [{}])[0].get("url")
            if use_memory:
                mem.write("cartographer", "fact", f"{key} how the activity proceeds: {question} -> {str(answer)[:400]}",
                          {"case": key, "url": url})
        except Exception:
            answer = None
    w = llm.ask_json(WRITE_ACT, f"Caption: {case['image_caption']}\nIn progress: {doing}\n"
                     f"Sourced fact about how it proceeds: {answer or '(none)'}", [image], max_tokens=220)
    text = (w.get("action") or "").strip()
    if text and use_memory:
        try:
            mem.write("director", "state", f"{key} action in progress: {text}", {"case": key, "url": url})
        except Exception:
            pass
    return {"text": text, "doing": doing, "question": question, "fact": None if answer is None else str(answer)[:300],
            "url": url, "source": "exa+llm" if answer else "llm"}


def make(case, image, use_web=True, use_memory=True):
    """Returns {"text": sentence or "", "entity": ..., "source": ..., "url": ..., "question": ...}."""
    key = case["task_id"]
    if use_memory:
        try:
            for h in mem.recall("composer", f"{key} state on return", 6):
                for text in [h.get("text") or ""] + list(h.get("excerpts") or []):
                    m = re.search(rf"{key} state on return: (.+)", text)
                    if m:
                        return {"text": m.group(1).strip(), "entity": None, "source": f"past.dev:{h.get('source_id')}",
                                "url": None, "question": None}
        except Exception:
            pass
    a = llm.ask_json(ASK, f"Caption: {case['image_caption']}", [image], max_tokens=300)
    entity, question, answer, url = a.get("entity"), a.get("question"), None, None
    if not entity or str(entity).lower() in ("null", "none", "nothing"):
        return {"text": "", "entity": None, "source": "static scene", "url": None, "question": None}
    if question and use_web:
        try:
            res = exa_search.answer(question)
            answer = res.get("answer")
            url = (res.get("citations") or [{}])[0].get("url")
            if use_memory:
                mem.write("cartographer", "fact", f"{key} rate of change: {question} -> {str(answer)[:400]}",
                          {"case": key, "url": url})
        except Exception:
            answer = None
    w = llm.ask_json(WRITE, f"Caption: {case['image_caption']}\nEntity: {entity}\nState now: {a.get('now')}\n"
                     f"Sourced fact about how fast it changes: {answer or '(none)'}", [image], max_tokens=200)
    later = (w.get("later") or "").strip()
    if later and use_memory:
        try:
            mem.write("director", "state", f"{key} state on return: {later}", {"case": key, "url": url})
        except Exception:
            pass
    return {"text": later, "entity": entity, "source": "exa+llm" if answer else "llm", "url": url,
            "question": question, "fact": None if answer is None else str(answer)[:300]}
