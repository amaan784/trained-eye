# Recording real clips

The experiments need real first-person footage for three skills, recorded on the
glasses. Training and validation clips must come from different sessions than the
test clips (a different day, angle or lighting), so nothing leaks between them.

## The three skills

| Skill | What to record | Near-misses to include |
|---|---|---|
| `washer_done` | The washer's front display through a full cycle end, and a while after it shows done | mid-cycle time remaining, display off, the dryer or another appliance |
| `boil_over` | A pot of water or pasta from simmering to foaming up near the rim (turn the heat down before it actually boils over) | gentle simmer, steam only, a different pot that is not boiling |
| `bus_m4` | The street where the M4 (or another route you can film) arrives, until it pulls in | other buses and routes, trucks, an empty street |

Keep people out of view where you can: film the display, the stove or the road, not
the room or the sidewalk crowd. Faces that do appear are blurred before anything is
published, and only footage you approve is ever published.

## Sessions

For each skill:

1. Session 1 and session 2 (train and validation): about 5 minutes in total. Look
   at the target the way you would in daily life, with natural head movement, and
   include the moment the condition becomes true.
2. Session 3 (test): about 5 minutes on a different day, or at least with a
   different angle and lighting. Include the near-misses above. Test labels are
   marked by you only (labeling tool in M3), never by a model.

The capture LED stays on the whole time; do not cover it.

## Files

Copy the clips to the Mac and name them like this:

    data/clips/<skill>/session1_<anything>.mp4
    data/clips/<skill>/session2_<anything>.mp4
    data/clips/<skill>/session3_test_<anything>.mp4

`data/` is gitignored, so clips are never committed by accident.

## The teaching clip

Separately, for the teach flow itself: one 10 second look at the target, plus the
spoken request recorded as audio (for example on the phone's voice memo app).

    data/teach/washer_done/teach.mp4
    data/teach/washer_done/request.m4a

Move your head a little during the look (closer, a step aside, a slight tilt): the
teach flow keeps 20 to 24 sharp, different keyframes and says so when it finds fewer than 20.
Video works best; a folder of frames must be JPEG, PNG or WebP (export iPhone HEIC photos
as JPEG first).

If training on model outputs is switched off (`TE_TRAIN_ON_MODEL_OUTPUTS=0`), the
dataset is only the keyframes you label yourself, and it needs both answers. Then record
the look so that the condition becomes true during it (the washer display changes to
End, the pot starts to boil) and label frames from before and after.
