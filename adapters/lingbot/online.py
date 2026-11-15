"""Online director and cartographer: planning that happens while the model generates.

Nothing here runs ahead of time. The first generation step looks at the scene once;
after that every step asks, for the heading the camera is about to face and the time
that has passed, what should be there, and feeds the answer into that step's
cross-attention. Lookups for the *next* heading and state run in background threads
while the GPU renders the current chunk, so generation does not wait on them.

Agents and what they do inside the loop (identities in shared memory, see tools/mem.py):
  director      looks at frames; owns the scene state (what the ongoing process looks
                like now), updates it as time passes, seen or unseen; writes each state
                to shared memory with its episode time
  cartographer  when the camera turns toward a direction nobody has described, searches
                the web (Exa) for what is really there, writes a visual description to
                shared memory, and it is used from the next step on
  composer      (this object, on behalf of the generator) recalls from shared memory
                what other agents know about this place and model, and binds each
                entry to its direction in the generator's cross-attention

The object duck-types `director.Timeline` so the episode runner can use either.
It never sees the benchmark rubric: only the first frame, caption, objective and
the frames the model itself has generated.
"""
import os
import re
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import exa_search  # noqa: E402
import llm  # noqa: E402
import mem  # noqa: E402
from director import facing_start  # noqa: E402

SECTORS = (-90, 90, 180)
SCENE_SYSTEM = """You direct a video world model. You see the first frame of a scene and its caption, and what the camera will do. Reply with one JSON object and nothing else:
{
 "place": name of the real, identifiable place if this is one (e.g. "Taj Mahal, Agra"), else null,
 "viewpoint": one sentence: where the camera stands and which way it faces,
 "process": one sentence naming the main ongoing process (who or what is doing what), or null if the scene is static,
 "rate": one short phrase: how fast that process visibly advances per 5 seconds, or null,
 "target_box": [x0,y0,x1,y1] normalised box (origin top left) covering ONLY the part of the image that will visibly change in the next few seconds (the blank area where the next mark, letter or item will appear, plus the hand or tool doing it), not the whole subject; keep it as small as you can; null if nothing is in progress,
 "subject": 40-70 words describing the first frame exactly as it is, in the caption's style,
 "away": 40-70 words describing what the camera sees when turned away from the first view: the surroundings only, same place, light and weather, continuing what is visible at the frame edges. It must NOT name or describe the subject of the first frame or anything that only exists in that view, and must not use the name of the place,
 "state": short label of the process state now (e.g. "one green stripe painted")
}
Lessons other agents have recorded about this world model (follow them):
%s"""
STATE_SYSTEM = """You keep the world state for a video world model. A process is under way in a scene. Given the first frame, the latest generated frame, the elapsed time and the last recorded state, decide what the process should look like NOW in the real world. Time passes whether or not the camera is looking: if the subject is out of view, advance the state by the elapsed time anyway. Progress is cumulative and causal: nothing un-happens. Keep every person, object, colour and position that does not change exactly as in the first frame. Reply with one JSON object and nothing else:
{"state": short label of how far the process has got now,
 "subject": 40-70 words describing the starting view as it should look NOW (same style as before, with the changed part stated concretely and visibly)}"""
SECTOR_SYSTEM = """You describe one direction of a real place for a video world model. From the search results, write what a camera at the stated viewpoint sees when facing the stated direction. Reply with one JSON object and nothing else:
{"visual": 15-30 words, purely visual (structures, materials, colours, vegetation, water, sky, light matching the stated time of day). No place names, no history, no numbers, no direction words. It must not describe the subject of the starting view or anything resembling it,
 "url": the URL of the search result that best supports it, or "none"}"""


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


class OnlineDirector:
    def __init__(self, case, first_frame, log=None, state_every=2, use_web=True, use_memory=True):
        self.case, self.first = case, first_frame
        self.log_fn, self.state_every = log, state_every
        self.use_web, self.use_memory = use_web, use_memory
        self.lock = threading.Lock()
        self.entries = {}          # sector -> {"heading","text","url","source"}
        self.pending = set()
        self.events = []           # everything the agents did, with episode time
        self.t0 = time.time()
        self.state_busy = False
        self.last_state_chunk = -99
        self.target_box = None
        self.spec = {"away_prompt": None, "sources": []}
        self._begin()

    # ---- bookkeeping ---------------------------------------------------------
    def _event(self, agent, what, **kw):
        e = dict(agent=agent, what=what, wall_s=round(time.time() - self.t0, 2), **kw)
        with self.lock:
            self.events.append(e)

    def _recall(self, query, limit=8):
        if not self.use_memory:
            return []
        t = time.time()
        try:
            hits = mem.recall("composer", query, limit)
        except Exception as e:
            self._event("composer", "recall failed", error=repr(e)[:120])
            return []
        self._event("composer", "recall", query=query, n=len(hits),
                    authors=sorted({h["author"] for h in hits if h.get("author")}),
                    seconds=round(time.time() - t, 2))
        return hits

    # ---- step 0: look at the scene once -----------------------------------------
    def _begin(self):
        lessons = [h["text"] for h in self._recall(
            "lessons about how this world model follows text prompts and camera turns", 6)
            if (h.get("kind") == "lesson" or "lesson" in (h.get("text") or ""))]
        sysmsg = SCENE_SYSTEM % ("\n".join("- " + x[:400] for x in lessons[:5]) or "- (none yet)")
        user = (f"Caption: {self.case['image_caption']}\nCamera objective: {self.case['prompt']}\n"
                f"Preset actions: {self.case['action']}")
        s = llm.ask_json(sysmsg, user, [self.first])
        self.scene = s
        self.key = _slug(s["place"]) if s.get("place") else self.case["task_id"]
        self.target_box = s.get("target_box") if s.get("process") else None
        self.spec["away_prompt"] = s["away"]
        self.state = {"from_s": 0, "state": s.get("state") or "start", "prompt": s["subject"]}
        self.states = [dict(self.state)]
        self._event("director", "scene", place=s.get("place"), process=s.get("process"),
                    state=self.state["state"], lessons_used=len(lessons))
        if s.get("place"):
            for sec in SECTORS:   # start filling in the other directions right away
                self._want(sec)

    # ---- cartographer: one direction, in the background -------------------------
    def _want(self, sector):
        with self.lock:
            if sector in self.entries or sector in self.pending:
                return
            self.pending.add(sector)
        threading.Thread(target=self._sector, args=(sector,), daemon=True).start()

    def _sector(self, sector):
        try:
            tag = f"{self.key} heading {sector}"
            for h in self._recall(f"{tag}: what is visible in that direction", 8):
                for text in [h.get("text") or ""] + list(h.get("excerpts") or []):
                    m = re.search(rf"{re.escape(tag)}: (.+)", text)
                    if m:
                        self._set_entry(sector, m.group(1).strip().rstrip("."), None,
                                        f"past.dev:{h.get('source_id')} by {h.get('author')}")
                        return
            if not self.use_web:
                return
            side = {-90: "to the viewer's left", 90: "to the viewer's right", 180: "directly behind the viewer"}[sector]
            q = f"{self.scene['place']} site layout: what is {side} when {self.scene['viewpoint']}"
            hits = exa_search.search(q, n=4)
            self._event("cartographer", "exa search", query=q, urls=[h["url"] for h in hits])
            out = llm.ask_json(SECTOR_SYSTEM, f"Place: {self.scene['place']}\nViewpoint: {self.scene['viewpoint']}\n"
                               f"Direction: {side}\nStarting view (do not describe): {self.scene['subject']}\n"
                               f"Search results:\n" + "\n".join(f"[{h['url']}] {h['highlights'][:500]}" for h in hits),
                               max_tokens=300)
            self._set_entry(sector, out["visual"].strip().rstrip("."), out.get("url"), "exa")
            if self.use_memory:
                mem.write("cartographer", "fact", f"{tag}: {out['visual'].strip()}",
                          {"scene": self.key, "heading": sector, "url": out.get("url"),
                           "case": self.case["task_id"]})
                self._event("cartographer", "wrote to memory", heading=sector, url=out.get("url"))
        except Exception as e:
            self._event("cartographer", "failed", heading=sector, error=repr(e)[:160])
        finally:
            with self.lock:
                self.pending.discard(sector)

    def _set_entry(self, sector, text, url, source):
        with self.lock:
            self.entries[sector] = {"heading": sector, "pitch": 0, "text": text, "url": url, "source": source}
            if url and url != "none":
                self.spec["sources"].append({"fact": text, "url": url})
        self._event("composer", "entry ready", heading=sector, source=source, text=text[:80])

    # ---- director: keep the state moving, in the background ---------------------
    def tick(self, t, chunk, latest_frame, in_view):
        """Called at the start of every generation step."""
        if not self.scene.get("process") or self.state_busy or chunk - self.last_state_chunk < self.state_every:
            return
        self.state_busy, self.last_state_chunk = True, chunk
        threading.Thread(target=self._advance, args=(t, latest_frame, in_view), daemon=True).start()

    def _advance(self, t, frame, in_view):
        try:
            user = (f"Process: {self.scene['process']}\nTypical rate: {self.scene.get('rate')}\n"
                    f"Elapsed: {t:.1f} s since the first frame.\nLast recorded state (at {self.state['from_s']:.1f} s): "
                    f"{self.state['state']}\nLast description: {self.state['prompt']}\n"
                    f"The subject is currently {'IN view (second image is the latest frame)' if in_view else 'OUT of view (second image shows where the camera is looking instead)'}.")
            out = llm.ask_json(STATE_SYSTEM, user, [self.first, frame], max_tokens=400)
            new = {"from_s": round(t, 2), "state": out["state"], "prompt": out["subject"]}
            with self.lock:
                self.state = new
                self.states.append(dict(new))
            self._event("director", "state", t=round(t, 2), state=out["state"], in_view=in_view)
            if self.use_memory:
                mem.write("director", "state", f"{self.key} at {t:.1f} s: {out['state']}",
                          {"scene": self.key, "case": self.case["task_id"], "episode_time_s": round(t, 2)})
        except Exception as e:
            self._event("director", "state failed", error=repr(e)[:160])
        finally:
            self.state_busy = False

    # ---- interface used by the episode runner (same as director.Timeline) -------
    def stage_at(self, t):
        with self.lock:
            return dict(self.state)

    def prompt_at(self, t, yaw=0.0, pitch=0.0):
        if not facing_start(yaw, pitch):
            return self.spec["away_prompt"]
        return self.stage_at(t)["prompt"]

    def target_mask(self, height, width, pad=0.04):
        if not self.target_box:
            return None
        x0, y0, x1, y1 = self.target_box
        m = np.zeros((height, width), np.uint8)
        m[int(max(0, y0 - pad) * height):int(min(1, y1 + pad) * height),
          int(max(0, x0 - pad) * width):int(min(1, x1 + pad) * width)] = 255
        return m

    def live_entries(self):
        with self.lock:
            return [self.entries[k] for k in sorted(self.entries)]

    def report(self):
        with self.lock:
            return {"scene": self.scene, "scene_key": self.key, "states": list(self.states),
                    "entries": [self.entries[k] for k in sorted(self.entries)],
                    "events": list(self.events)}
