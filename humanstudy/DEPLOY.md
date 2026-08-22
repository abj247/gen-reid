# Viewing and deploying the study

Two separate things: looking at it yourself while it runs on the cluster, and putting it
somewhere the public can reach.

## 1. Viewing it now, from your own machine

The application runs on a compute node with no route from the outside world. Reach it with an SSH
tunnel: your laptop forwards a local port through the login node to the compute node.

```bash
# On YOUR machine, not the cluster. Replace <login-host> with the host you normally ssh into.
ssh -N -L 8811:c3-4:8811 ab260989@<login-host>
```

Leave that running, then open:

- `http://localhost:8811/` for the participant flow
- `http://localhost:8811/author?key=devkey` for author mode

`c3-4` is whichever compute node the app is on; `hostname` on the cluster prints it. If the login
node cannot reach the compute node directly, hop through it:

```bash
ssh -N -J ab260989@<login-host> -L 8811:localhost:8811 ab260989@c3-4
```

To restart the app on the cluster:

```bash
cd ~/gen-reid/humanstudy
AUTHOR_KEY=devkey ./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8811
```

This is for looking at it. It is not a way to run the study: the moment your SSH session ends the
tunnel dies, and a compute node allocation is not a stable address.

## 2. Making it public

Three free services, each doing one job. The split is not arbitrary: it is what the constraints
force.

| What | Service | Why not the obvious alternative |
|---|---|---|
| Application | Render, free web service | Fly.io's free allowance ended; Railway has no free tier |
| Database | Neon, free Postgres | Render's free Postgres expires after 30 days, and free web services have no persistent disk, so SQLite is lost on restart |
| Video | Cloudflare R2 | R2 charges nothing for egress. Serving video from the app burns a small egress allowance within a few hundred sessions |

### Step 1: database

Create a project at neon.tech and copy the connection string. That is `DATABASE_URL`. The schema is
created automatically on first boot.

### Step 2: video

Create an R2 bucket, then upload the prepared files:

```bash
cd ~/gen-reid/humanstudy
# Any S3-compatible client works; rclone and the aws CLI both do.
aws s3 sync data/video/ s3://<bucket>/persistqa/ \
    --endpoint-url https://<account-id>.r2.cloudflarestorage.com \
    --content-type video/mp4
```

Enable public read access on the bucket, or attach a custom domain, and set `VIDEO_BASE_URL` to the
resulting base URL. The files are named by opaque media id, so nothing in a URL identifies a source
video.

### Step 3: application

Point Render at this repository with root directory `humanstudy`, environment Docker. Set:

```
DATABASE_URL          the Neon connection string
VIDEO_BASE_URL        the R2 base URL, no trailing slash
AUTHOR_KEY            a long random string, not "devkey"
AUTHOR_NAMES          your two names, comma separated
COOKIE_SECURE         1
PERSISTQA_POOL_SALT   must match the value used when the pools were built
```

`PERSISTQA_POOL_SALT` is the one that fails confusingly if you get it wrong: the media ids will not
match the uploaded filenames and every video will 404. If you did not set it when running
`build_pools.py`, the default in that file is the value to use.

### Step 4: check before sharing the link

```bash
curl -s https://<your-app>.onrender.com/healthz
```

Then complete one session yourself end to end, on a phone as well as a laptop, and confirm the video
plays and seeks. Confirm `/author` returns 404 without the key.

## What to expect from the free tier

The service sleeps after 15 minutes idle, and the next request takes about a minute. A participant
who lands on a spinner will leave. Two mitigations, in order of value: keep the tab warm during any
recruitment push with an uptime pinger hitting `/healthz` every 10 minutes, and accept the cold
start otherwise, because all state lives in Postgres so a sleep mid-session costs latency only and
never data.

## Two things to settle before the link goes out

**Copyright.** The videos are film footage from CinePile, MovieChat-1k and LVU. The repository avoids
redistribution by not shipping videos; a public link does redistribute them. That is a question about
those corpora's licences, and it is worth answering before rather than after.

**Consent.** There is a one-line data notice, not a consent gate, which is defensible for an
anonymous study collecting no personal data. If this goes through an ethics board or a paid
participant panel, they will expect a consent screen. It is a small change to the landing template.
