# Question construction

The pipeline that produced the released question set. These scripts are here for provenance and
audit. Running them is not required to use the benchmark, and rerunning them will not reproduce
the released file byte for byte, because several stages call hosted language models.

## Order of operations

1. **Identifier assignment.** `assign_ids.py` maps source-corpus video identifiers onto the
   anonymous identifier space and writes the mapping released as
   `../data/video_id_mapping.json`.

2. **Question generation.** Questions are written against dense per-video annotation. Each names a
   referent by visual description at one moment and asks about an attribute of that same referent
   at a different moment, which is the property that forces identity to survive a shot change.

3. **Option construction.** `generate_debiased_options.py` produces eight options per question and
   balances the correct letter across positions, so a model cannot gain by preferring one letter.

4. **Adversarial debiasing.** `debias_via_committee.py` is the stage that matters most. A committee
   of thirteen language models answers every question with no video. Any question the committee
   beats is removed or repaired, and the cycle repeats. Multiple-choice video questions leak
   language priors, and a benchmark that does not measure and report its own text-only accuracy
   cannot claim its numbers reflect vision. After filtering, text-only accuracy sits at the chance
   rate.

5. **Metadata tagging.** `llm_retag_reid.py` assigns the capability axis and the ten-way
   identity-challenge axis. Both are defined in `../data/README.md`.

`debias_report.json` records what the committee removed and why.

## What to check if you regenerate

Any regenerated set should be validated on three properties before use, because each has failed at
least once during development:

- Text-only accuracy at or near the chance rate. If it is above, the filtering did not converge.
- Correct letters balanced across the eight positions.
- Option lengths distributed similarly across positions, so that length is not a cue for the
  correct answer.
