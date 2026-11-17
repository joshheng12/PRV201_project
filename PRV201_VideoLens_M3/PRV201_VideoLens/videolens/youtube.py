"""Collect public metadata for explicitly supplied IDs. Does not download videos."""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import pandas as pd

def duration_seconds(value):
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", value)
    if not match or not any(match.groups()):
        raise ValueError("Unsupported video duration")
    h,m,s = (float(x or 0) for x in match.groups())
    return h*3600 + m*60 + s

def collect_videos(ids_path, output_dir):
    key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not key:
        raise ValueError("Set YOUTUBE_API_KEY in .env")
    ids = list(dict.fromkeys(x.strip() for x in Path(ids_path).read_text().splitlines() if x.strip()))
    if not 1 <= len(ids) <= 200 or any(not re.fullmatch(r"[A-Za-z0-9_-]{11}", x) for x in ids):
        raise ValueError("Provide 1-200 YouTube IDs, exactly 11 characters each")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Collection output directory must be empty")
    rows, excluded, raw = [], [], []
    found = set()
    for start in range(0, len(ids), 50):
        query = urlencode({"part": "snippet,statistics,contentDetails", "id": ",".join(ids[start:start+50]), "key": key})
        try:
            with urlopen("https://www.googleapis.com/youtube/v3/videos?" + query, timeout=30) as response:
                payload = json.load(response)
        except Exception as exc:
            raise RuntimeError("YouTube request failed; check key, quota, and network. Partial collection is not saved.") from None
        raw.append(payload)
        for item in payload.get("items", []):
            vid = item["id"]
            found.add(vid)
            stats, snippet = item.get("statistics", {}), item["snippet"]
            missing = [k for k in ["viewCount", "likeCount", "commentCount"] if k not in stats]
            if missing:
                excluded.append({"video_id": vid, "reason": "Missing public metrics: " + ", ".join(missing)})
                continue
            try:
                seconds = duration_seconds(item["contentDetails"]["duration"])
                if seconds <= 0 or int(stats["viewCount"]) <= 0:
                    raise ValueError("Nonpositive duration/views")
            except ValueError:
                excluded.append({"video_id": vid, "reason": "Nonpositive or unsupported duration/views"})
                continue
            hashtags = " ".join(re.findall(r"#\w+", snippet.get("title", "") + " " + snippet.get("description", "")))
            rows.append({"video_id": vid, "platform": "youtube", "title": snippet["title"],
                "description": snippet.get("description", ""), "views": int(stats["viewCount"]),
                "likes": int(stats["likeCount"]), "comments": int(stats["commentCount"]),
                "duration_seconds": seconds, "hashtags": hashtags, "data_origin": "observed",
                "creator_id": snippet.get("channelId", ""), "published_at": snippet.get("publishedAt", ""),
                "collected_at": datetime.now(timezone.utc).isoformat()})
    excluded += [{"video_id": vid, "reason": "Unavailable/private/deleted/not returned"} for vid in ids if vid not in found]
    pd.DataFrame(rows).to_csv(out / "videos.csv", index=False)
    pd.DataFrame(excluded, columns=["video_id", "reason"]).to_csv(out / "excluded.csv", index=False)
    (out / "raw_api_responses.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return {"requested": len(ids), "collected": len(rows), "excluded": len(excluded), "output": str(out)}
