"""
Telegram output sink for broadcasting transcription results.

Runs as a dedicated thread, consuming messages from a queue
and forwarding them to a Telegram chat via the Bot API.
"""

import queue
import time
import datetime
import requests

# Max Telegram message length
_TG_MAX_LEN = 4096

# Rate-limit guard
_MIN_SEND_INTERVAL = 0.05  # 50 ms between sends


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def telegram_sink_worker(
    tg_q: "queue.Queue",
    *,
    bot_token: str = "",
    chat_id: str = "",
) -> None:
    """
    Blocking worker for Telegram message sending.

    Run in a daemon thread. Items in queue should be dicts with "text" key,
    or the string "STOP" to terminate.

    Args:
        tg_q: Queue to consume messages from
        bot_token: Telegram bot token from @BotFather
        chat_id: Target Telegram chat/group/channel ID
    """
    if not bot_token or not chat_id:
        _log("[TG] bot_token or chat_id not set => Telegram disabled.")
        while True:
            item = tg_q.get()
            if item == "STOP":
                return

    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    _log(f"[TG] Telegram sink started (chat_id={chat_id}).")

    last_send = 0.0

    while True:
        try:
            item = tg_q.get(timeout=1.0)
        except queue.Empty:
            continue

        if item == "STOP":
            break

        text = (item.get("text") or "").strip() if isinstance(item, dict) else ""
        if not text:
            continue

        # Split into chunks that fit Telegram's limit
        for i in range(0, len(text), _TG_MAX_LEN):
            chunk = text[i : i + _TG_MAX_LEN]

            # Respect minimum interval between sends
            elapsed = time.time() - last_send
            if elapsed < _MIN_SEND_INTERVAL:
                time.sleep(_MIN_SEND_INTERVAL - elapsed)

            try:
                resp = requests.post(
                    api_url,
                    json={
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=10,
                )
                if resp.status_code == 429:
                    try:
                        retry_after = resp.json().get("parameters", {}).get("retry_after", 1)
                    except Exception:
                        retry_after = 1
                    _log(f"[TG] Rate limited. Sleeping {retry_after}s.")
                    time.sleep(float(retry_after))
                elif not resp.ok:
                    _log(f"[TG] API {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                _log(f"[TG] Send error: {e}")

            last_send = time.time()

    _log("[TG] Telegram sink stopped.")
