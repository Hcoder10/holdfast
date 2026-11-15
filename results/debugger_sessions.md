# Debugger sessions (debug1, 15 Nov, 6 cases)

Each case starts from the unmodified model (no writes). `adapters/lingbot/debugger.py` runs it, reads the critic's verdict and a state report, recalls earlier sessions from past.dev, patches the state program, and reruns, up to three attempts. Scores are the in-loop critic's (Claude Haiku on frames, 0 to 8), not the benchmark judge's.

| Case | Request | Critic score, unmodified | Best after debugging | Program kept |
|---|---|---|---|---|
| GC002 | Move along a rectangular path around the central full-body sculpture w | 5.0 | 8.0 | `{"anchor": 0.8, "shadow": true, "landmarks": true, "cam_gain": 3}` |
| GC010 | Turn right 90 degrees, then turn back left to the starting view | 5.0 | 8.0 | `{"shadow": true, "landmarks": true, "cam_gain": 3}` |
| GC025 | Walk into the room, then walk back out, then walk in again | 6.5 | 6.5 | `none (no patch beat the unmodified run)` |
| IF011 | The woman walks forward to the base of the Great Sphinx of Egypt | 3.0 | 5.0 | `{"register_gain": 16, "cam_gain": 3}` |
| OE005 | Wait and observe. | 4.0 | 4.0 | `none (no patch beat the unmodified run)` |
| OE043 | Move the camera left 5 steps, then move back to the starting point. | 5.0 | 5.0 | `none (no patch beat the unmodified run)` |

## GC002: Move along a rectangular path around the central full-body sculpture with the camera held facing the original 

- **Attempt 0**: critic 5.0 (action 0, subject 2, picture 1, alive 1). The camera does not execute the requested rectangular path around the central sculpture; instead it shows static or minimal movement with repeated similar viewpoints, and the lighting transitions from
  - State report: frame-to-frame motion by third of the video: 5.5, 6.7, 6.2 (below 2 is nearly frozen); mean colour shift first to last frame: 23.3 (above 12 is visible drift); last frame vs first frame difference: 47.1 (below 20 means the video ends on the starting view)
  - Recalled from past.dev: 5 notes, e.g. "controller: In the OE043 debug session, the camera did not move left or return to the starting point as requested (problem)."
  - Symptom named: camera does not execute rectangular path, static frames. Hypothesis: The camera movement command is not being registered with sufficient gain, and the view is not being anchored to prevent colour drift and ensure return to start position.
  - Patch applied: `{"anchor": 0.8, "shadow": true, "landmarks": true, "cam_gain": 3}`
- **Attempt 1**: critic 8.0 (action 2, subject 2, picture 1, alive 1). The camera completes a rectangular path around the central statue and returns to origin as requested, but the crowd of figures in the background appears largely static and frozen throughout the sequen
  - State report: frame-to-frame motion by third of the video: 11.8, 8.0, 12.4 (below 2 is nearly frozen); mean colour shift first to last frame: 17.0 (above 12 is visible drift); last frame vs first frame difference: 21.5 (below 20 means the video ends on the starting view); state pins on return: 1 (forks 0, dropped 0, landmarks {'kept': 4, 'pins': 1})
  - Recalled from past.dev: 6 notes, e.g. "[program from controller] debug session on OE005 (first person; request: Wait and observe.): symptom 'scene frozen, no frame-to-frame motion'; patch {"cam_gain": 2}; crit"
  - Symptom named: background figures frozen, incomplete return to origin. Hypothesis: The landmark pinning is too restrictive (only 1 pin retained, forks dropped); increasing landmark tolerance and age threshold will allow better state capture during the rectangular path and smoother return to origin.
  - Patch applied: `{"landmarks": true}`
- **Attempt 2**: critic 8.0 (action 2, subject 2, picture 1, alive 1). The camera completes a rectangular path around the central sculpture and returns to the original position as requested, but the scene exhibits noticeable color shifting (warming to reddish tones in la
  - State report: frame-to-frame motion by third of the video: 11.8, 8.0, 12.4 (below 2 is nearly frozen); mean colour shift first to last frame: 17.0 (above 12 is visible drift); last frame vs first frame difference: 21.5 (below 20 means the video ends on the starting view); state pins on return: 1 (forks 0, dropped 0, landmarks {'kept': 4, 'pins': 1})
- Kept: attempt 1.

## GC010: Turn right 90 degrees, then turn back left to the starting view

- **Attempt 0**: critic 5.0 (action 0, subject 2, picture 1, alive 1). Camera motion requested (turn right 90°, then left back to start) does not occur; viewpoint remains stationary with only zooming and framing adjustments instead of the required camera rotation.
  - State report: frame-to-frame motion by third of the video: 20.1, 26.8, 22.4 (below 2 is nearly frozen); mean colour shift first to last frame: 36.1 (above 12 is visible drift); last frame vs first frame difference: 56.1 (below 20 means the video ends on the starting view); camera turns commanded +0 deg, measured -17 deg over 8 chunks
  - Recalled from past.dev: 3 notes, e.g. "controller: In the OE043 debug session, the camera did not move left or return to the starting point as requested (problem)."
  - Symptom named: camera turn not executed first person. Hypothesis: The camera rotation command is not being properly registered and executed; raising cam_gain will amplify the turning signal in the first-person viewpoint controller.
  - Patch applied: `{"shadow": true, "landmarks": true, "cam_gain": 3}`
- **Attempt 1**: critic 8.0 (action 2, subject 2, picture 1, alive 1). Camera motion appears to execute the requested 90-degree right turn and return, but some frames show repetition and the scene exhibits noticeable color inconsistency across the sequence.
  - State report: frame-to-frame motion by third of the video: 28.8, 37.0, 27.6 (below 2 is nearly frozen); mean colour shift first to last frame: 56.0 (above 12 is visible drift); last frame vs first frame difference: 68.1 (below 20 means the video ends on the starting view); camera turns commanded +0 deg, measured -448 deg over 8 chunks; state pins on return: 1 (forks 2, dropped 0, landmarks {'kept': 1, 'pins': 0})
  - Recalled from past.dev: 6 notes, e.g. "[program from controller] debug session on GC002 (first person; request: Move along a rectangular path around the central full-body sculpture with the camera held ): symp"
  - Symptom named: camera turn not executed, color drift persists. Hypothesis: The measured camera turn of -448 degrees indicates the turn is happening in the latent state but not properly manifesting in output frames; increasing anchor further and disabling landmarks (which may be interfering with
  - Patch applied: `{"anchor": 0.8, "shadow": true, "landmarks": false, "cam_gain": 3}`
- **Attempt 2**: critic 8.0 (action 2, subject 2, picture 1, alive 1). Camera motion appears to execute the requested 90-degree right turn and return, but some frames show repetitive positioning and the scene exhibits visible compression artifacts and slight color incons
  - State report: frame-to-frame motion by third of the video: 34.9, 37.7, 28.0 (below 2 is nearly frozen); mean colour shift first to last frame: 38.9 (above 12 is visible drift); last frame vs first frame difference: 57.4 (below 20 means the video ends on the starting view); camera turns commanded +0 deg, measured +314 deg over 8 chunks; state pins on return: 1 (forks 2, dropped 0, landmarks None)
- Kept: attempt 1.

## GC025: Walk into the room, then walk back out, then walk in again

- **Attempt 0**: critic 6.5 (action 1, subject 2, picture 1, alive 1). The requested walk-in, walk-out, walk-in sequence is only partially executed with incomplete transitions and the viewpoint never fully exits the room.
  - State report: frame-to-frame motion by third of the video: 8.6, 11.8, 10.8 (below 2 is nearly frozen); mean colour shift first to last frame: 9.7 (above 12 is visible drift); last frame vs first frame difference: 47.8 (below 20 means the video ends on the starting view)
  - Recalled from past.dev: 6 notes, e.g. "[program from controller] debug session on OE043 (first person; request: Move the camera left 5 steps, then move back to the starting point.): symptom 'camera does not mo"
  - Symptom named: incomplete walk sequence, never fully exits. Hypothesis: First-person shadow and landmarks patches are needed to maintain the doorway view on return, allowing proper tracking of the exit and re-entry sequence.
  - Patch applied: `{"shadow": true, "landmarks": true, "cam_gain": 3}`
- **Attempt 1**: critic 3.5 (action 0, subject 1, picture 1, alive 1). The requested action (walk in, walk out, walk in again) is not clearly executed; the viewpoint mostly remains stationary or shows repeated similar angles without distinct entrance/exit transitions.
  - State report: frame-to-frame motion by third of the video: 11.0, 14.3, 11.7 (below 2 is nearly frozen); mean colour shift first to last frame: 9.1 (above 12 is visible drift); last frame vs first frame difference: 47.9 (below 20 means the video ends on the starting view); state pins on return: 2 (forks 0, dropped 0, landmarks {'kept': 2, 'pins': 2})
  - Recalled from past.dev: 4 notes, e.g. "[program from controller] debug session on OE043 (first person; request: Move the camera left 5 steps, then move back to the starting point.): symptom 'camera does not mo"
  - Symptom named: viewpoint does not fully exit room. Hypothesis: The shadow and landmarks parameters from attempt 1 are anchoring the camera too strongly to its starting position, preventing a clean exit; increasing cam_gain further will override this over-anchoring and allow the full
  - Patch applied: `{"shadow": false, "landmarks": false, "cam_gain": 3}`
- **Attempt 2**: critic 1.0 (action 0, subject 0, picture 1, alive 0). The video shows multiple disconnected hallway/doorway perspectives rather than a coherent first-person walk into a room, out, and back in again.
  - State report: frame-to-frame motion by third of the video: 11.0, 14.0, 16.6 (below 2 is nearly frozen); mean colour shift first to last frame: 12.7 (above 12 is visible drift); last frame vs first frame difference: 49.0 (below 20 means the video ends on the starting view)
- Kept: attempt 0.

## IF011: The woman walks forward to the base of the Great Sphinx of Egypt

- **Attempt 0**: critic 3.0 (action 0, subject 2, picture 0, alive 0). The video is a static repetition of identical frames with no forward movement toward the Sphinx; the scene appears completely frozen with severe tiling/duplication artifacts.
  - State report: frame-to-frame motion by third of the video: 2.7, 2.5, 2.3 (below 2 is nearly frozen); mean colour shift first to last frame: 0.3 (above 12 is visible drift); last frame vs first frame difference: 10.9 (below 20 means the video ends on the starting view)
  - Recalled from past.dev: 0 notes
  - Symptom named: static frames with no forward movement. Hypothesis: The model is not receiving clear instruction to move the subject forward; raising register_gain will amplify attention to the walking request, and increasing cam_gain will ensure the forward motion toward the Sphinx is e
  - Patch applied: `{"register_gain": 16, "cam_gain": 3}`
- **Attempt 1**: critic 5.0 (action 0, subject 2, picture 1, alive 1). The woman does not walk forward toward the Great Sphinx; instead she remains stationary or moves laterally across the frame, failing to carry out the requested action.
  - State report: frame-to-frame motion by third of the video: 2.8, 3.9, 4.3 (below 2 is nearly frozen); mean colour shift first to last frame: 1.8 (above 12 is visible drift); last frame vs first frame difference: 22.2 (below 20 means the video ends on the starting view); request register gain 16 (1000 attention calls biased)
  - Recalled from past.dev: 0 notes
  - Symptom named: woman does not walk forward toward Sphinx. Hypothesis: The register gain of 16 is too high and causing the model to over-attend to the request text at the expense of coherent motion generation; reducing it will allow the world model to balance request following with natural 
  - Patch applied: `{"register_gain": 8, "cam_gain": 2}`
- **Attempt 2**: critic 5.0 (action 0, subject 2, picture 1, alive 1). The woman does not walk forward; she remains in a static or nearly identical position across all frames, failing to approach the sphinx as requested.
  - State report: frame-to-frame motion by third of the video: 3.1, 3.0, 2.8 (below 2 is nearly frozen); mean colour shift first to last frame: 0.7 (above 12 is visible drift); last frame vs first frame difference: 10.1 (below 20 means the video ends on the starting view); request register gain 8 (2000 attention calls biased)
- Kept: attempt 1.

## OE005: Wait and observe.

- **Attempt 0**: critic 4.0 (action 0, subject 2, picture 1, alive 0). Scene is completely static across all frames with no evolution or movement, failing the 'wait and observe' directive.
  - State report: frame-to-frame motion by third of the video: 5.7, 5.2, 5.6 (below 2 is nearly frozen); mean colour shift first to last frame: 5.9 (above 12 is visible drift); last frame vs first frame difference: 36.1 (below 20 means the video ends on the starting view)
  - Recalled from past.dev: 6 notes, e.g. "controller: The debug session on IF011 had static frames with no forward movement (problem)."
  - Symptom named: scene frozen, no frame-to-frame motion. Hypothesis: First-person viewpoint with 'wait and observe' request needs shadow and landmarks enabled to anchor the observer position and track the scene evolution, similar to the OE043 camera movement case.
  - Patch applied: `{"cam_gain": 2}`
- **Attempt 1**: critic 4.0 (action 0, subject 2, picture 1, alive 0). The scene is completely frozen across all frames with no meaningful evolution or movement despite the request to 'wait and observe.'
  - State report: frame-to-frame motion by third of the video: 5.7, 5.2, 5.6 (below 2 is nearly frozen); mean colour shift first to last frame: 5.9 (above 12 is visible drift); last frame vs first frame difference: 36.1 (below 20 means the video ends on the starting view)
  - Recalled from past.dev: 5 notes, e.g. "controller: The debug session on IF011 had static frames with no forward movement (problem)."
  - Symptom named: scene completely static no frame evolution. Hypothesis: First-person 'wait and observe' requires landmarks and shadow to maintain viewpoint memory while scene_action drives subject motion; without these, the model has no directive to animate the children/ball.
  - Patch applied: `{"scene_action": "timeline", "cam_gain": 1}`
- **Attempt 2**: critic 1.0 (action 0, subject 0, picture 1, alive 0). The scene is essentially frozen with no meaningful evolution across frames; children and ball remain in identical positions, and multiple frames show duplicate or near-identical compositions suggestin
  - State report: frame-to-frame motion by third of the video: 6.2, 6.6, 7.0 (below 2 is nearly frozen); mean colour shift first to last frame: 4.9 (above 12 is visible drift); last frame vs first frame difference: 39.3 (below 20 means the video ends on the starting view)
- Kept: attempt 0.

## OE043: Move the camera left 5 steps, then move back to the starting point.

- **Attempt 0**: critic 5.0 (action 0, subject 2, picture 1, alive 1). Camera does not move left or return to starting point; frames show repetitive slight variations of the same scene rather than the requested leftward pan and return.
  - State report: frame-to-frame motion by third of the video: 11.1, 10.9, 12.7 (below 2 is nearly frozen); mean colour shift first to last frame: 3.4 (above 12 is visible drift); last frame vs first frame difference: 36.4 (below 20 means the video ends on the starting view)
  - Recalled from past.dev: 1 notes, e.g. "[program from controller] debug session on IF011 (third person; request: The woman walks forward to the base of the Great Sphinx of Egypt): symptom 'static frames with no"
  - Symptom named: camera does not move left or return. Hypothesis: The model needs higher camera gain to execute the requested leftward pan and return movement, and shadow/landmarks flags to maintain view consistency for the return to starting point.
  - Patch applied: `{"shadow": true, "landmarks": true, "cam_gain": 3}`
- **Attempt 1**: critic 5.0 (action 0, subject 2, picture 1, alive 1). No visible camera motion occurs; the viewpoint remains static throughout instead of moving left 5 steps and returning.
  - State report: frame-to-frame motion by third of the video: 10.1, 15.1, 19.0 (below 2 is nearly frozen); mean colour shift first to last frame: 20.9 (above 12 is visible drift); last frame vs first frame difference: 26.1 (below 20 means the video ends on the starting view); state pins on return: 1 (forks 0, dropped 0, landmarks {'kept': 1, 'pins': 1})
  - Recalled from past.dev: 3 notes, e.g. "[program from controller] debug session on IF011 (third person; request: The woman walks forward to the base of the Great Sphinx of Egypt): symptom 'static frames with no"
  - Symptom named: camera does not move left or return. Hypothesis: The cam_gain of 3.0 is insufficient to drive the requested leftward camera motion; increasing it should amplify the camera movement command execution in the attention mechanism.
  - Patch applied: `{"shadow": true, "landmarks": true, "cam_gain": 3}`
- **Attempt 2**: critic 5.0 (action 0, subject 2, picture 1, alive 1). Camera does not move left 5 steps or return to starting point; the viewpoint remains static throughout the sequence.
  - State report: frame-to-frame motion by third of the video: 10.1, 15.1, 19.0 (below 2 is nearly frozen); mean colour shift first to last frame: 20.9 (above 12 is visible drift); last frame vs first frame difference: 26.1 (below 20 means the video ends on the starting view); state pins on return: 1 (forks 0, dropped 0, landmarks {'kept': 1, 'pins': 1})
- Kept: attempt 0.
