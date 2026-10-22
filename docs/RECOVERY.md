# Interrupted-chat recovery

A gateway restart during a turn can leave that turn marked unknown. Hermes pauses later inputs until an operator acknowledges it. This is separate from model memory admission.

On the machine running Hermes, use the upstream command with the affected session ID or title:

```text
hermes sessions discard SESSION_ID --yes
```

This acknowledges unfinished turns; it does not delete the conversation or replay their input. Queued inputs can then proceed. The Desktop queue also exposes Discard for an interrupted-turn card.

The model request queue independently quarantines requests with uncertain transport outcomes. Its release check requires the same owned router to confirm idle, explicitly unloaded, or an exact authenticated model-not-found response. Authentication failures, malformed replies and changed router identities keep the conservative guard.
