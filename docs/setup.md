# Setup

Accounts, keys and Modal Secrets are in [../SETUP.md](../SETUP.md). Run everything from the
repository root.

    brew install python@3.12 ffmpeg          # python3 may be Apple's 3.9; ffmpeg reads video
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,train]"
    cp .env.example .env                     # then fill in the keys

After `pip install -e .`, `te <command>` is the same as `python -m backend.cli <command>`.
`te --help` lists every command, and [../CLAUDE.md](../CLAUDE.md) has them with a line
each.

Check the install, offline and free:

    pytest
    ruff check .

Face blurring for anything published needs OpenCV 4 (OpenCV 5's headless build has no
Haar face detector):

    pip install "opencv-python-headless<5"

The README art is rendered with headless Chrome:

    python -m backend.art.build
