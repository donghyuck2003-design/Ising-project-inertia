from __future__ import annotations
from pathlib import Path
import subprocess
from typing import Optional

DEFAULT_SEND_MAIL_SCRIPT = Path('/home/onion120/mail/send_mail.sh')


def send_completion_email(
    recipient: Optional[str],
    subject: str,
    body: str,
    *,
    script_info: str,
    send_mail_script: str | Path = DEFAULT_SEND_MAIL_SCRIPT,
) -> bool:
    """Send a non-fatal completion email through the user's mail helper.

    Notification failure never changes the experiment's original status.
    """
    if not recipient:
        return False
    helper = Path(send_mail_script).expanduser()
    if not helper.is_file() or not helper.stat().st_mode & 0o111:
        print(f'Warning: mail helper is not executable: {helper}', flush=True)
        return False
    try:
        proc = subprocess.run(
            [
                str(helper),
                '--to', str(recipient),
                '--subject', str(subject),
                '--body', str(body),
                '--script-info', str(script_info),
            ],
            check=False,
        )
        if proc.returncode != 0:
            print(f'Warning: completion email helper returned {proc.returncode}', flush=True)
            return False
        return True
    except Exception as exc:
        print(f'Warning: completion email could not be sent: {exc}', flush=True)
        return False
