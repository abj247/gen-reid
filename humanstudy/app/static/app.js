/* PersistQA Human Benchmarking: client behaviour.
 *
 * Three responsibilities, deliberately kept apart:
 *   PQA.telemetry()  buffered event reporting that survives a dropped connection
 *   PQA.watchGate()  unlock the questions once the video has been watched
 *   PQA.question()   keyboard control, time on question, answer-change counting
 *
 * Answers never travel through the telemetry path. An answer is a normal form POST that
 * the participant waits on; telemetry is best effort and is allowed to fail silently.
 * Mixing the two would let a flaky network turn a lost measurement into a lost answer.
 *
 * No framework and no build step. The whole interface is a video element and a radio
 * group, and a bundler would add a toolchain this repository does not otherwise have.
 */
(function () {
  "use strict";

  var PQA = {};
  var buffer = [];
  var seq = 0;
  var sessionId = null;
  var timer = null;

  /* ------------------------------------------------------------------ telemetry */

  function push(kind, extra) {
    if (!sessionId) return;
    var e = { kind: kind, seq: seq++, ts: Date.now() };
    if (extra) {
      if (extra.position_s !== undefined) e.position_s = extra.position_s;
      if (extra.value !== undefined) e.value = String(extra.value);
    }
    buffer.push(e);
    // Flush on a full batch so a long session never accumulates an unbounded buffer.
    if (buffer.length >= 20) flush(false);
  }

  function flush(useBeacon) {
    if (!sessionId || !buffer.length) return;
    var payload = JSON.stringify({ events: buffer });
    // Events are only cleared once handed off. The server deduplicates on (session, seq),
    // so a batch that is sent twice because a response was lost costs nothing.
    buffer = [];
    try {
      if (useBeacon && navigator.sendBeacon) {
        // A Blob with an explicit type is required; a bare string is sent as
        // text/plain and some servers reject it.
        navigator.sendBeacon("/telemetry", new Blob([payload], { type: "application/json" }));
      } else {
        fetch("/telemetry", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload,
          keepalive: true,
        }).catch(function () { /* best effort by design */ });
      }
    } catch (err) { /* never let reporting break the page */ }
  }

  PQA.telemetry = function (id) {
    sessionId = id;
    if (timer) clearInterval(timer);
    timer = setInterval(function () { flush(false); }, 10000);

    // pagehide rather than unload: unload is unreliable on mobile Safari, where a tab
    // switch may never fire it and the session's tail would be lost.
    window.addEventListener("pagehide", function () { flush(true); });
    document.addEventListener("visibilitychange", function () {
      push(document.hidden ? "hide" : "show");
      if (document.hidden) flush(true);
    });

    var v = document.getElementById("player");
    if (!v) return;
    var lastSeek = null;
    ["play", "pause", "ended"].forEach(function (kind) {
      v.addEventListener(kind, function () { push(kind, { position_s: v.currentTime }); });
    });
    v.addEventListener("seeking", function () { lastSeek = v.currentTime; });
    v.addEventListener("seeked", function () {
      push("seek", { position_s: v.currentTime, value: lastSeek === null ? "" : lastSeek.toFixed(1) });
    });
    v.addEventListener("error", function () { push("error", { value: "media" }); });
  };

  /* ------------------------------------------------------------------ watch gate */

  PQA.watchGate = function (alreadyWatched) {
    var v = document.getElementById("player");
    var go = document.getElementById("go");
    var hint = document.getElementById("hint");
    var err = document.getElementById("loaderr");
    if (!v || !go) return;

    function unlock() {
      go.hidden = false;
      if (hint) hint.textContent = "You can replay or scrub the video while answering.";
      fetch("/watch/complete", { method: "POST" }).catch(function () {});
    }

    if (alreadyWatched) unlock();
    v.addEventListener("ended", unlock);

    // Reaching the end is the intent, not literally firing `ended`: a participant who
    // scrubs to the last second and stops would otherwise be stuck with no way forward.
    v.addEventListener("timeupdate", function () {
      if (v.duration && v.currentTime >= v.duration - 1.5) unlock();
    });

    v.addEventListener("error", function () { if (err) err.hidden = false; });
  };

  /* -------------------------------------------------------------------- question */

  PQA.question = function (position, nOptions) {
    var form = document.getElementById("qform");
    var next = document.getElementById("next");
    if (!form) return;

    var radios = Array.prototype.slice.call(form.querySelectorAll('input[name="chosen"]'));
    var started = Date.now();
    var changes = 0;
    var firstChoice = null;

    push("question_view", { value: String(position) });

    radios.forEach(function (r) {
      r.addEventListener("change", function () {
        // The first selection is not a change; only revisions after it are. Counting the
        // first would make every answered question look revised and the measure useless.
        if (firstChoice === null) firstChoice = r.value;
        else if (r.value !== firstChoice) changes += 1;
        if (next) next.disabled = false;
      });
    });

    form.addEventListener("submit", function () {
      var ms = document.getElementById("ms_on_question");
      var nc = document.getElementById("n_changes");
      if (ms) ms.value = String(Date.now() - started);
      if (nc) nc.value = String(changes);
      flush(false);
    });

    PQA.go = function (action) {
      var a = document.getElementById("action");
      if (a) a.value = action;
      // Back must not be blocked by the disabled Next button, and must still record the
      // time already spent, so it submits the form rather than navigating away.
      if (action === "back") {
        var chosen = form.querySelector('input[name="chosen"]:checked');
        if (!chosen && radios.length) radios[0].checked = true;
      }
      form.requestSubmit ? form.requestSubmit() : form.submit();
    };

    document.addEventListener("keydown", function (ev) {
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      var tag = (ev.target.tagName || "").toLowerCase();
      if (tag === "input" && ev.target.type === "text") return;

      // 1..8 select the nth option. Letters are not bound: the option letters are A..H
      // and binding them would collide with typing in any future free-text field.
      var n = parseInt(ev.key, 10);
      if (n >= 1 && n <= Math.min(nOptions, 9) && radios[n - 1]) {
        radios[n - 1].checked = true;
        radios[n - 1].dispatchEvent(new Event("change", { bubbles: true }));
        ev.preventDefault();
        return;
      }
      if (ev.key === "Enter" && next && !next.disabled) {
        PQA.go("next");
        ev.preventDefault();
      }
    });
  };

  window.PQA = PQA;
})();
