# PersistQA Human Benchmarking

A web application that collects human answers to PersistQA questions, so that model accuracy can be
compared against human accuracy on exactly the same questions.

It is deliberately isolated from the rest of the repository. It imports nothing from `benchmark/` or
`solutions/`, has no machine learning dependencies, reads one frozen JSON file, and deploys on its
own. That isolation is what lets it run on a free hosting tier and start in seconds.

## Two modes

**Public.** A participant is assigned one video and answers all of that video's questions, which is
twelve to eighteen depending on the video. One video per participant, because nobody outside the
project will watch several long videos. Sessions run about fifteen to twenty minutes.

**Author.** Access-restricted. An author works through a larger set across as many sittings as they
like, resuming where they stopped. Progress is saved after every question.

The two pools overlap on purpose. Where an author and a member of the public answered the same
video, the two groups can be compared directly, which is the cheapest available check that the
public responses are not noise.

## Layout

```
prepare/build_pools.py    selects the videos and freezes them to data/pools.json
prepare/encode_videos.py  prepares the videos for browser streaming
app/db.py                 schema: participants, sessions, responses, telemetry
app/assignment.py         pool loading and balanced video assignment
app/main.py               routes
app/templates/            server-rendered pages, no build step
app/static/               one stylesheet, one script
```

## Running it locally

Requires Python 3.10 or newer. The repository's research environment is not used and must not be.

```bash
cd humanstudy
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# Regenerate the pools from the benchmark. Needs the benchmark data and the video corpus.
python prepare/build_pools.py

# Prepare the videos. Mostly a lossless remux; see the note below.
python prepare/encode_videos.py

# Serve. Author mode is unreachable unless AUTHOR_KEY is set.
AUTHOR_KEY=devkey ./.venv/bin/uvicorn app.main:app --port 8811
```

Then open `http://127.0.0.1:8811/`, or `http://127.0.0.1:8811/author?key=devkey`.

With no `DATABASE_URL` the app uses a local SQLite file, which is fine for development and wrong for
deployment.

## How videos are selected

`build_pools.py` applies four filters and prints what each one dropped:

1. **Merged by real video.** Nine real videos appear under two or three anonymous entries. Those
   entries overlap rather than partition, so one video's rows can repeat the same question text
   verbatim; the worst case collapses 44 rows to 19 distinct questions. Questions are deduplicated
   by text so no participant is ever shown the same question twice.
2. **At least twelve questions**, and at most eighteen for the public pool. A video with three
   questions is not worth watching; a video with forty-four is a forty-minute session and would be
   abandoned. Oversized videos go to the author pool, where session length does not matter.
3. **Human referent only.** Any video whose question or option text mentions an animal is dropped.
   The word list is deliberately over-eager: a false positive costs one video, a false negative puts
   an animal question in front of a participant.
4. **The file exists and has a model baseline.** Videos the evaluation stack cannot decode are
   excluded benchmark-wide, and a video with no baseline has no measurable difficulty.

Difficulty is measured, not assumed: it is the accuracy the evaluated backbones achieved on that
video at a uniform eight-frame budget.

## Two things that are easy to get wrong

**Videos are addressed by an opaque id.** The real video ids are YouTube ids, and hiding them is the
entire purpose of the benchmark's anonymous id space. Every video is served under an HMAC-derived
`media_id`, so no URL, page source, or network log a participant can see maps back to a source
video. `PERSISTQA_POOL_SALT` must match between `build_pools.py` and the running app or media URLs
will not resolve.

**Most videos are remuxed, not re-encoded.** The instinct is to transcode everything to a uniform
480p. On this corpus that is wrong: the sources are already 640x360 H.264 at a modest bitrate, so a
480p pass upscales them and produces more bytes while losing a generation of quality. Measured, that
pass produced 128 percent of the original size. The default is a stream copy that only relocates the
moov atom, which is lossless and still fixes the seeking problem participants actually hit. Only
oversized or oversized-resolution files are re-encoded.

## Deployment

The application must not serve video. Free hosting tiers meter egress tightly, and at roughly eight
megabytes per session a small allowance is gone within a few hundred participants. Point
`VIDEO_BASE_URL` at object storage with cheap or zero egress and upload the prepared files there.

Free tiers also have no persistent disk, so a SQLite file is lost on restart. Set `DATABASE_URL` to a
managed Postgres.

Configuration is in `.env.example`. The variables that change behaviour rather than merely wiring
things up are `VIDEO_BASE_URL`, `DATABASE_URL`, `AUTHOR_KEY` and `PERSISTQA_POOL_SALT`.

## What is recorded

Answers, the time spent on each question, whether an answer was changed before submitting, and
player events including play, pause, seek and completion. No names, no email addresses, no IP
addresses. The landing page states this in one line.

The telemetry exists to support quality filtering after collection: sessions that never played the
video, or that answered implausibly fast, can be identified and reported separately. Nothing is
rejected at collection time, and abandoned sessions are kept rather than deleted. Discarding the
sessions people walked away from would preferentially discard the hard videos and inflate the human
accuracy the study exists to measure.

There is no consent gate, only a data notice. That was a deliberate choice for an anonymous study
collecting no personal data. If this is ever run through an ethics board or a paid participant
panel, a short consent screen is expected and is a small change to the landing page.
