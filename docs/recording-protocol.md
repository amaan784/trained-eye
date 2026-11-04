# Recording real clips

The experiments need real first-person footage for three kitchen skills, recorded on the
glasses, plus a fourth if you record it. Training and validation clips must come from
different sessions than the test clips (a different day, angle or lighting), so nothing
leaks between them.

## The skills

Each skill has a hold: watch mode alerts only once the condition has stayed true that
long, so opening the fridge for a moment says nothing.

| Skill | Hold | What to record | Near-misses to include |
|---|---|---|---|
| `burner_left_on` | 10 s | A gas or electric burner lit (a flame, or a glowing ring) with no pan on it, or a pan nobody is tending, for well over 10 s | the burner off; a burner lit for a few seconds and turned off; a pan being stirred on a lit burner; the oven light or another appliance's light |
| `fridge_left_open` | 30 s | The fridge door standing open, from the moment it opens until well past 30 s | the door closed; opened for 10 s and shut; the freezer or a cupboard door open |
| `tap_left_running` | 20 s | The kitchen tap running into the sink with nobody using it, for well over 20 s | the tap off; filling a glass or rinsing a cup for a few seconds; water dripping; the dishwasher running |
| `egg_whisked` (only if you record it) | 0 s | Eggs or egg whites being whisked in a bowl until they are foamy (or form peaks), and a while after | unbeaten eggs; eggs half whisked; a bowl of milk or batter |

The hold shapes the clips:

- Keep each true stretch going well past the skill's hold (aim for at least twice it), and
  keep filming a while after: the evaluation expects the alert at the end of the hold.
- Record short ones under the hold too (a fridge opened for 10 s, a tap used for a few
  seconds, a burner lit briefly). They are the near-misses the hold exists for: nothing
  should fire for them.

Safety first. Never leave a burner on unattended to film it: stay at the stove the whole
time, keep nothing that can catch fire near it, and turn it off when you stop. The burner
skill is a reminder, never a safety device; it does not replace a smoke alarm. Turn the
tap off when you are done, and do not leave the fridge open longer than the clip needs.

Keep people out of view where you can: film the stove, the fridge or the sink, not the
room. Faces that do appear are blurred before anything is published, and only footage
you approve is ever published.

## Sessions

For each skill:

1. Session 1 and session 2 (train and validation): about 5 minutes in total. Look at the
   target the way you would in daily life, with natural head movement, and include the
   moment the condition becomes true and a stretch past its hold. Session 2 sets the
   alert threshold (`te calibrate`), so give it near-misses too.
2. Session 3 (test): about 5 minutes on a different day, or at least with a different
   angle and lighting. Include the near-misses above. Test labels are marked by you only
   (`te label --clip`), never by a model.

The capture LED stays on the whole time; do not cover it.

## Files

Copy the clips to the Mac and name them like this:

    data/clips/<skill>/session1_<anything>.mp4
    data/clips/<skill>/session2_<anything>.mp4
    data/clips/<skill>/session3_test_<anything>.mp4

`data/` is gitignored, so clips are never committed by accident. Label each clip with
`te label --clip <clip>`: mark the time ranges where the condition itself is true (the
door open, the water running), not when the alert should come. The hold is added by watch
mode and the evaluation.

## The teaching clip

Separately, for the teach flow itself: one 10 second look at the target, plus the
spoken request recorded as audio (for example on the phone's voice memo app).

    data/teach/burner_left_on/teach.mp4
    data/teach/burner_left_on/request.m4a

For example: "Tell me when a burner has been left on", "Tell me when the fridge door is
left open", "Tell me when the tap is left running", "Tell me when the egg is whisked to
a foam".

Move your head a little during the look (closer, a step aside, a slight tilt): the
teach flow keeps 20 to 24 sharp, different keyframes and says so when it finds fewer than 20.
Video works best; a folder of frames must be JPEG, PNG or WebP (export iPhone HEIC photos
as JPEG first).

If training on model outputs is switched off (`TE_TRAIN_ON_MODEL_OUTPUTS=0`), the
dataset is only the keyframes you label yourself, and it needs both answers. Then record
the look so that the condition becomes true during it (the burner lights, the fridge door
opens, the tap turns on, the eggs turn foamy) and label frames from before and after.
