"""PersistQA Human Benchmarking: application routes.

Flow
----
public   /  ->  /watch  ->  /q/0 .. /q/N-1  ->  /review  ->  /done
author   /author (dashboard)  ->  the same watch and question flow, many videos

Session identity is a single opaque cookie holding the session's resume token. The token
is 128 bits of randomness generated server side and is the only thing the client holds,
so there is nothing to forge and no signing secret to manage. A participant returning
within the cookie lifetime lands on their first unanswered question rather than starting
again.

Every answer write is an upsert keyed on (session, question). A double click, a refresh
or a retried request updates the existing row instead of creating a second one, which is
what keeps the response count honest.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as OrmSession

from .assignment import choose_video, load_pools, question_index, question_order, video_index
from .db import Participant, Response as StudyResponse, Session as StudySession, TelemetryEvent
from .db import SessionLocal, init_db, utcnow

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

# Where the browser fetches video from. In deployment this points at object storage with
# cheap egress; locally it falls back to this app serving the encoded files directly,
# which is fine for one developer and unacceptable for a public link.
VIDEO_BASE_URL = os.environ.get("VIDEO_BASE_URL", "").rstrip("/")
LOCAL_VIDEO_DIR = BASE_DIR / "data" / "video"

# Author mode is gated by a shared secret rather than accounts. Two people need access on
# their own machines; a password database would be more surface area for no benefit.
AUTHOR_KEY = os.environ.get("AUTHOR_KEY", "")
AUTHOR_NAMES = [n for n in os.environ.get("AUTHOR_NAMES", "author-1,author-2").split(",") if n]

COOKIE = "pqa_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 14  # two weeks, so an author can resume across sittings

app = FastAPI(title="PersistQA Human Benchmarking", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@app.on_event("startup")
def _startup() -> None:
    init_db()
    load_pools()  # fail fast at boot if the pool file is missing or empty


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- helpers


def video_url(video_id: str) -> str:
    """Public URL for a video, addressed by its opaque media id.

    The real video ids are YouTube ids, which is exactly what the benchmark's anonymous
    id space exists to hide. Addressing media by the HMAC-derived id means no URL, page
    source, or network log seen by a participant can be mapped back to a source video.
    """
    media_id = video_index()[video_id]["media_id"]
    if VIDEO_BASE_URL:
        return f"{VIDEO_BASE_URL}/{media_id}.mp4"
    return f"/video/{media_id}.mp4"


def current_session(request: Request, db: OrmSession) -> StudySession | None:
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    return db.scalar(select(StudySession).where(StudySession.resume_token == token))


def set_session_cookie(response: Response, session: StudySession) -> None:
    response.set_cookie(
        COOKIE, session.resume_token, max_age=COOKIE_MAX_AGE,
        httponly=True, samesite="lax", secure=bool(os.environ.get("COOKIE_SECURE")),
    )


def keys_of(session: StudySession) -> list[str]:
    return json.loads(session.question_keys)


def answered_map(db: OrmSession, session: StudySession) -> dict[str, StudyResponse]:
    rows = db.scalars(
        select(StudyResponse).where(StudyResponse.session_id == session.id)
    ).all()
    return {r.question_key: r for r in rows}


def first_unanswered(db: OrmSession, session: StudySession) -> int:
    answered = answered_map(db, session)
    for i, key in enumerate(keys_of(session)):
        r = answered.get(key)
        if r is None or r.chosen is None:
            return i
    return len(keys_of(session))


def start_session(db: OrmSession, mode: str, participant: Participant,
                  exclude: set[str] | None = None) -> StudySession | None:
    video = choose_video(db, mode, exclude=exclude)
    if video is None:
        return None
    session = StudySession(participant_id=participant.id, mode=mode, video_id=video,
                           question_keys="[]")
    db.add(session)
    db.flush()  # need the generated id to seed the question order
    session.question_keys = json.dumps(question_order(video, session.id))
    db.commit()
    return session


# ---------------------------------------------------------------------------- public


@app.get("/", response_class=HTMLResponse)
def landing(request: Request, db: OrmSession = Depends(get_db)):
    session = current_session(request, db)
    if session and session.status == "in_progress":
        return templates.TemplateResponse("landing.html", {
            "request": request, "resuming": True,
            "n_questions": len(keys_of(session)),
        })
    return templates.TemplateResponse("landing.html", {"request": request, "resuming": False})


@app.post("/start")
def start(request: Request, db: OrmSession = Depends(get_db)):
    session = current_session(request, db)
    if session and session.status == "in_progress":
        return RedirectResponse("/watch", status_code=303)

    participant = Participant(mode="public", user_agent=request.headers.get("user-agent", "")[:400])
    db.add(participant)
    db.flush()
    session = start_session(db, "public", participant)
    if session is None:
        raise HTTPException(503, "No videos are available right now.")
    response = RedirectResponse("/watch", status_code=303)
    set_session_cookie(response, session)
    return response


@app.get("/watch", response_class=HTMLResponse)
def watch(request: Request, db: OrmSession = Depends(get_db)):
    session = current_session(request, db)
    if session is None:
        return RedirectResponse("/", status_code=303)
    if session.status == "completed":
        return RedirectResponse("/done", status_code=303)
    record = video_index()[session.video_id]
    return templates.TemplateResponse("watch.html", {
        "request": request, "session": session,
        "video_url": video_url(session.video_id),
        "n_questions": len(keys_of(session)),
        "already_watched": session.video_completed,
        "resume_at": first_unanswered(db, session),
    })


@app.post("/watch/complete")
def watch_complete(request: Request, db: OrmSession = Depends(get_db)):
    session = current_session(request, db)
    if session is None:
        raise HTTPException(404)
    session.video_completed = True
    db.commit()
    return JSONResponse({"ok": True})


@app.get("/q/{position}", response_class=HTMLResponse)
def question(position: int, request: Request, db: OrmSession = Depends(get_db)):
    session = current_session(request, db)
    if session is None:
        return RedirectResponse("/", status_code=303)
    if session.status == "completed":
        return RedirectResponse("/done", status_code=303)
    keys = keys_of(session)
    if position < 0 or position >= len(keys):
        return RedirectResponse("/review", status_code=303)

    key = keys[position]
    question_record = question_index()[key]
    existing = answered_map(db, session).get(key)

    # Record that the question was displayed, so time on question can be reconstructed
    # even if the participant leaves without answering.
    if existing is None:
        db.add(StudyResponse(session_id=session.id, question_key=key, position=position,
                             correct=question_record["correct_answer"]))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()

    return templates.TemplateResponse("question.html", {
        "request": request, "session": session,
        "video_url": video_url(session.video_id),
        "q": question_record, "position": position, "total": len(keys),
        "chosen": existing.chosen if existing else None,
        "options": sorted(question_record["options"].items()),
    })


@app.post("/q/{position}")
def answer(position: int, request: Request, chosen: str = Form(...),
           ms_on_question: int = Form(0), n_changes: int = Form(0),
           action: str = Form("next"), db: OrmSession = Depends(get_db)):
    session = current_session(request, db)
    if session is None:
        return RedirectResponse("/", status_code=303)
    keys = keys_of(session)
    if position < 0 or position >= len(keys):
        raise HTTPException(400, "position out of range")

    key = keys[position]
    question_record = question_index()[key]
    chosen = (chosen or "").strip().upper()[:1]
    if chosen not in question_record["options"]:
        raise HTTPException(400, "invalid option")

    row = answered_map(db, session).get(key)
    if row is None:
        row = StudyResponse(session_id=session.id, question_key=key, position=position,
                            correct=question_record["correct_answer"])
        db.add(row)
    row.chosen = chosen
    row.is_correct = chosen == question_record["correct_answer"]
    # Accumulate rather than overwrite: a participant who returns to a question has spent
    # time on it more than once and both visits are part of the effort measurement.
    row.ms_on_question = (row.ms_on_question or 0) + max(0, ms_on_question)
    row.n_answer_changes = (row.n_answer_changes or 0) + max(0, n_changes)
    row.answered_at = utcnow()
    db.commit()

    if action == "back":
        return RedirectResponse(f"/q/{max(0, position - 1)}", status_code=303)
    if position + 1 >= len(keys):
        return RedirectResponse("/review", status_code=303)
    return RedirectResponse(f"/q/{position + 1}", status_code=303)


@app.get("/review", response_class=HTMLResponse)
def review(request: Request, db: OrmSession = Depends(get_db)):
    session = current_session(request, db)
    if session is None:
        return RedirectResponse("/", status_code=303)
    if session.status == "completed":
        return RedirectResponse("/done", status_code=303)
    answered = answered_map(db, session)
    items = []
    for i, key in enumerate(keys_of(session)):
        row = answered.get(key)
        items.append({
            "position": i,
            "text": question_index()[key]["question_text"],
            "chosen": row.chosen if row else None,
        })
    return templates.TemplateResponse("review.html", {
        "request": request, "items": items,
        "n_unanswered": sum(1 for it in items if not it["chosen"]),
    })


@app.post("/submit")
def submit(request: Request, db: OrmSession = Depends(get_db)):
    session = current_session(request, db)
    if session is None:
        return RedirectResponse("/", status_code=303)
    session.status = "completed"
    session.completed_at = utcnow()
    db.commit()
    return RedirectResponse("/done", status_code=303)


@app.get("/done", response_class=HTMLResponse)
def done(request: Request, db: OrmSession = Depends(get_db)):
    session = current_session(request, db)
    if session is None:
        return RedirectResponse("/", status_code=303)
    rows = [r for r in answered_map(db, session).values() if r.chosen]
    correct = sum(1 for r in rows if r.is_correct)
    is_author = session.mode == "author"
    return templates.TemplateResponse("done.html", {
        "request": request, "correct": correct, "answered": len(rows),
        "total": len(keys_of(session)), "is_author": is_author,
    })


# ------------------------------------------------------------------------- telemetry


@app.post("/telemetry")
async def telemetry(request: Request, db: OrmSession = Depends(get_db)):
    """Accept a batch of player events.

    Always returns success. Telemetry is diagnostic; a malformed or duplicated batch must
    never surface an error to a participant who is trying to answer a question, and must
    never be able to roll back an answer.
    """
    session = current_session(request, db)
    if session is None:
        return JSONResponse({"ok": True})
    try:
        events = (await request.json()).get("events", [])
    except Exception:
        return JSONResponse({"ok": True})

    for event in events[:200]:
        try:
            db.add(TelemetryEvent(
                session_id=session.id,
                kind=str(event.get("kind", ""))[:24],
                position_s=float(event["position_s"]) if event.get("position_s") is not None else None,
                value=str(event.get("value"))[:120] if event.get("value") is not None else None,
                client_seq=int(event["seq"]),
                client_ts=float(event["ts"]),
            ))
            db.commit()
        except (IntegrityError, KeyError, TypeError, ValueError):
            # IntegrityError is the normal path for a re-sent batch, not a problem.
            db.rollback()
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------- author


@app.get("/author", response_class=HTMLResponse)
def author_home(request: Request, key: str = "", db: OrmSession = Depends(get_db)):
    if not AUTHOR_KEY or key != AUTHOR_KEY:
        raise HTTPException(404)
    pools = load_pools()
    videos = pools["author"]
    session = current_session(request, db)
    name = session.participant.author_name if session and session.mode == "author" else None

    progress = {}
    if name:
        rows = db.scalars(
            select(StudySession).join(Participant)
            .where(Participant.author_name == name, StudySession.mode == "author")
        ).all()
        for s in rows:
            answered = sum(1 for r in s.responses if r.chosen)
            progress[s.video_id] = {
                "answered": answered, "total": len(keys_of(s)), "status": s.status,
            }
    return templates.TemplateResponse("author.html", {
        "request": request, "key": key, "names": AUTHOR_NAMES, "name": name,
        "videos": videos, "progress": progress,
        "total_questions": sum(v["n_questions"] for v in videos),
    })


@app.post("/author/start")
def author_start(request: Request, key: str = Form(...), name: str = Form(...),
                 db: OrmSession = Depends(get_db)):
    if not AUTHOR_KEY or key != AUTHOR_KEY:
        raise HTTPException(404)
    if name not in AUTHOR_NAMES:
        raise HTTPException(400, "unknown author")

    # Resume an open session for this author before assigning anything new.
    open_session = db.scalar(
        select(StudySession).join(Participant)
        .where(Participant.author_name == name, StudySession.mode == "author",
               StudySession.status == "in_progress")
        .order_by(StudySession.started_at)
    )
    if open_session is None:
        seen = set(db.scalars(
            select(StudySession.video_id).join(Participant)
            .where(Participant.author_name == name, StudySession.mode == "author")
        ).all())
        participant = Participant(mode="author", author_name=name,
                                  user_agent=request.headers.get("user-agent", "")[:400])
        db.add(participant)
        db.flush()
        open_session = start_session(db, "author", participant, exclude=seen)
        if open_session is None:
            return RedirectResponse(f"/author?key={key}", status_code=303)

    response = RedirectResponse("/watch", status_code=303)
    set_session_cookie(response, open_session)
    return response


# ----------------------------------------------------------------------- local video


@app.get("/video/{filename}")
def local_video(filename: str):
    """Serve encoded video locally.

    Development convenience only. In deployment VIDEO_BASE_URL points at object storage,
    because serving video from the app would exhaust the hosting tier's egress allowance
    within a few hundred sessions.
    """
    if VIDEO_BASE_URL:
        raise HTTPException(404)
    path = (LOCAL_VIDEO_DIR / filename).resolve()
    if not str(path).startswith(str(LOCAL_VIDEO_DIR.resolve())) or not path.is_file():
        raise HTTPException(404)
    from fastapi.responses import FileResponse
    return FileResponse(path, media_type="video/mp4")


@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}
