#!/usr/bin/env python3
import sys
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def rename_events(input_path, output_path, old_tags, new_tag):
    writer = EventFileWriter(str(output_path))

    for ev in event_file_loader.EventFileLoader(str(input_path)).Load():
        if ev.summary:
            for v in ev.summary.value:
                if v.tag in old_tags:
                    v.tag = new_tag
        writer.add_event(ev)

    writer.close()


def rename_events_dir(input_dir, output_dir, old_tags, new_tag):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ev_file in input_dir.glob("**/*.tfevents*"):
        out_file = Path(output_dir, ev_file.relative_to(input_dir))
        out_file.parent.mkdir(parents=True, exist_ok=True)
        rename_events(ev_file, out_file, old_tags, new_tag)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"{sys.argv[0]} <input dir> <output dir> <old tags> <new tag>")
        sys.exit(1)

    input_dir, output_dir, old_tags, new_tag = sys.argv[1:]
    old_tags = old_tags.split(";")

    rename_events_dir(input_dir, output_dir, old_tags, new_tag)
    print("Done")
