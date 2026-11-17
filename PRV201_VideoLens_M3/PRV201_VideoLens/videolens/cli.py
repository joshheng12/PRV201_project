import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from .data import make_demo
from .pipeline import run
from .providers import PROVIDERS

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="VideoLens / PRV-201 reproducible analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("make-demo", help="Generate explicitly synthetic fixtures")
    demo.add_argument("--out", default="data")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--training", default="data/seed_training.csv")
    analyze.add_argument("--torch", action="store_true", help="Execute PyTorch neural comparison")
    analyze.add_argument("--frames", help="Root with VIDEO_ID/*.jpg or *.png (max three per video)")
    analyze.add_argument("--providers", nargs="*", choices=PROVIDERS, default=[])
    analyze.add_argument("--reviews", help="Human review CSV")
    collect = commands.add_parser("collect-youtube")
    collect.add_argument("--ids", required=True, help="Text file with one video ID per line; max 200")
    collect.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.command == "make-demo":
            make_demo(args.out)
            print("Created SYNTHETIC demo fixtures; not real research data.")
        elif args.command == "collect-youtube":
            from .youtube import collect_videos
            print(json.dumps(collect_videos(args.ids, args.out), indent=2))
        else:
            stats = run(args.input, args.out, args.training, args.torch, args.frames, args.providers, args.reviews)
            print(stats["computed_summary"])
            print("Report:", str((Path(args.out) / "report.html").resolve()))
    except Exception as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")

if __name__ == "__main__":
    main()
