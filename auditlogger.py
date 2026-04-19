import json
import hashlib
from datetime import datetime


class AuditLogger:
    def __init__(self, path="auditlog.jsonl"):
        self.path = path
        self.prev_hash = "0" * 64

    def log(self, event_type, data):
        entry = {
            "time": datetime.utcnow().isoformat() + "Z",
            "event": event_type,
            "data": data,
            "prev_hash": self.prev_hash,
        }
        raw = json.dumps(entry, sort_keys=True).encode("utf-8")
        entry["hash"] = hashlib.sha256(raw).hexdigest()

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        self.prev_hash = entry["hash"]
