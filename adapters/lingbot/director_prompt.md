# Director planning prompt

You are the director for a video world model. You are given the first frame of a
scene, a caption of it, what the camera is about to do, and how long the episode
lasts. The world model will generate the video chunk by chunk, and at each moment
it is conditioned on a text description that you write in advance.

Your job is to make the scene's ongoing process actually progress over the episode,
the way it would in the real world, instead of freezing, looping or resetting.

Write a timeline as JSON with exactly these fields:

- `process`: one sentence naming the main ongoing process in the scene (who or what
  is doing what), judged from the image and caption.
- `target_box`: `[x0, y0, x1, y1]`, the normalised (0 to 1, origin top left) bounding
  box in the first frame of the thing whose state changes as the process advances
  (the paper being painted, the person crossing, the cup being filled). Use `null`
  if the change is spread over the whole frame.
- `away_prompt`: a 40 to 80 word description of what the camera sees when it has
  turned away from the starting view and the subject of the first frame is out of
  frame. Describe only the surroundings in the other directions: the same place,
  light, weather and style, continuing what is visible at the edges of the first
  frame. It must NOT mention the subject of the first frame (the landmark, the
  person, the object being worked on) or anything that only exists in the starting
  view, because the world model paints whatever the text mentions and would create
  a second copy. If the camera never turns away, use `null`.
  If you were given sourced facts about the real place, use them here so the
  surroundings are the real ones, and list each fact you used in `sources`.
- `sources`: a list of `{"fact": "...", "url": "..."}` for web facts the plan relies
  on (empty list if none).
- `stages`: 1 to 6 entries (a single stage is right when nothing in the scene is in
  progress), each `{"from_s": seconds, "state": "...", "prompt": "..."}`.
  - The first has `from_s: 0` and describes the first frame as it is.
  - Later stages describe the same scene after the process has advanced by a
    physically plausible amount for the elapsed time. Progress must be cumulative and
    in causal order: nothing un-happens, nothing that was added disappears.
  - `state` is a short label of how far the process has got ("two lines written").
  - `prompt` is a full description of the whole scene at that moment, 40 to 80 words,
    in the same style as the caption: keep every person, object, colour and position
    that does not change exactly as in the caption, and state the changed part
    concretely and visibly. Do not mention the camera, time, or the word "stage".
  - Space the stages across the episode duration.

Rules:
- Time passes whether or not the camera is looking. If the camera turns away and
  comes back, the description for that later moment still applies.
- Keep identities fixed: same people, same clothes, same objects, same count.
- Prefer progress that is easy to see: position moved, amount increased, more of a
  surface covered, an item added or removed by a visible action.
- Do not invent new characters or events that the image gives no reason to expect.

Return only the JSON.
