"""Peel spec for stage v22 (handover and setup guide), on top of v21."""

BASE, FIRST, LAST = 21, 22, 22

PEEL = {
    22: {"remove": ["HANDOVER.md", "SETUP.md"], "restore": ["CLAUDE.md", "PROGRESS.md"]},
}
