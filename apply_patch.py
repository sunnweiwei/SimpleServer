#!/usr/bin/env python3

"""
A self-contained **pure-Python 3.9+** utility for applying human-readable
“pseudo-diff” patch files to a collection of text files.
"""

from __future__ import annotations

import pathlib
import re
import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)


# --------------------------------------------------------------------------- #
#  Domain objects
# --------------------------------------------------------------------------- #
class ActionType(str, Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


@dataclass
class FileChange:
    type: ActionType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    move_path: Optional[str] = None


@dataclass
class Commit:
    changes: Dict[str, FileChange] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Exceptions
# --------------------------------------------------------------------------- #
class DiffError(ValueError):
    """Any problem detected while parsing or applying a patch."""


# --------------------------------------------------------------------------- #
#  Normalization & helper for fuzzy errors
# --------------------------------------------------------------------------- #
def canonical(s: str) -> str:
    """
    Create a normalized form of the line for matching:
    - strip CR
    - convert tabs to spaces
    - remove escaping backslashes
    - collapse whitespace to single spaces
    - strip ends
    """
    text = s.replace("\r", "").replace("\t", " ")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
#  Helper dataclasses used while parsing patches
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    orig_index: int = -1
    del_lines: List[str] = field(default_factory=list)
    ins_lines: List[str] = field(default_factory=list)


@dataclass
class PatchAction:
    type: ActionType
    new_file: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    move_path: Optional[str] = None


@dataclass
class Patch:
    actions: Dict[str, PatchAction] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Patch text parser
# --------------------------------------------------------------------------- #
@dataclass
class Parser:
    current_files: Dict[str, str]
    lines: List[str]
    index: int = 0
    patch: Patch = field(default_factory=Patch)
    fuzz: int = 0

    def _cur_line(self) -> str:
        if self.index >= len(self.lines):
            raise DiffError("Unexpected end of input while parsing patch")
        return self.lines[self.index]

    @staticmethod
    def _norm(line: str) -> str:
        return line.rstrip("\r")

    def is_done(self, prefixes: Optional[Tuple[str, ...]] = None) -> bool:
        if self.index >= len(self.lines):
            return True
        if prefixes and self._norm(self._cur_line()).startswith(prefixes):
            return True
        return False

    def startswith(self, prefix: Union[str, Tuple[str, ...]]) -> bool:
        return self._norm(self._cur_line()).startswith(prefix)

    def read_str(self, prefix: str) -> str:
        if prefix == "":
            raise ValueError("read_str() requires a non-empty prefix")
        if self._norm(self._cur_line()).startswith(prefix):
            text = self._cur_line()[len(prefix):]
            self.index += 1
            return text
        return ""

    def read_line(self) -> str:
        line = self._cur_line()
        self.index += 1
        return line

    def parse(self) -> None:
        while not self.is_done(("*** End Patch",)):
            # Update
            path = self.read_str("*** Update File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Duplicate update for file: {path}")
                move_to = self.read_str("*** Move to: ") or None
                if path not in self.current_files:
                    raise DiffError(f"Update File Error - missing file: {path}")
                action = self._parse_update_file(self.current_files[path])
                action.move_path = move_to
                self.patch.actions[path] = action
                continue

            # Delete
            path = self.read_str("*** Delete File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Duplicate delete for file: {path}")
                if path not in self.current_files:
                    raise DiffError(f"Delete File Error - missing file: {path}")
                self.patch.actions[path] = PatchAction(type=ActionType.DELETE)
                continue

            # Add
            path = self.read_str("*** Add File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Duplicate add for file: {path}")
                if path in self.current_files:
                    raise DiffError(f"Add File Error - file already exists: {path}")
                self.patch.actions[path] = self._parse_add_file()
                continue

            raise DiffError(f"Unknown line while parsing: {self._cur_line()}")

        if not self.startswith("*** End Patch"):
            raise DiffError("Missing *** End Patch sentinel")
        self.index += 1

    def _parse_update_file(self, text: str) -> PatchAction:
        action = PatchAction(type=ActionType.UPDATE)
        lines = text.split("\n")
        idx = 0

        while not self.is_done((
            "*** End Patch",
            "*** Update File:",
            "*** Delete File:",
            "*** Add File:",
            "*** End of File",
        )):
            def_str = self.read_str("@@ ")
            section_str = ""
            if not def_str and self._norm(self._cur_line()) == "@@":
                section_str = self.read_line()

            if not (def_str or section_str or idx == 0):
                raise DiffError(f"Invalid line in update section:\n{self._cur_line()}")

            next_ctx, chunks, end_idx, eof = peek_next_section(self.lines, self.index)
            new_idx, fuzz = find_context(self.lines_for_context(lines), next_ctx, idx, eof)
            if new_idx == -1:
                # rich error: show expected snippet and best fuzzy candidate
                target = canonical(next_ctx[0])
                best_ratio, best_i = 0.0, -1
                for i, ln in enumerate(lines):
                    r = difflib.SequenceMatcher(None, canonical(ln), target).ratio()
                    if r > best_ratio:
                        best_ratio, best_i = r, i
                snippet = "\n".join(next_ctx[:3]) + ("\n…" if len(next_ctx) > 3 else "")
                actual = lines[best_i] if best_i >= 0 else ""
                raise DiffError(
                    f"Context match failed at patch offset {self.index}.\n"
                    f"Expected snippet:\n>>> {snippet}\n"
                    f"Closest match in file at line {best_i+1} "
                    f"(similarity {best_ratio:.2f}):\n>>> {actual!r}"
                )
            self.fuzz += fuzz
            for ch in chunks:
                ch.orig_index += new_idx
                action.chunks.append(ch)
            idx = new_idx + len(next_ctx)
            self.index = end_idx

        return action

    def _parse_add_file(self) -> PatchAction:
        lines: List[str] = []
        while not self.is_done((
            "*** End Patch",
            "*** Update File:",
            "*** Delete File:",
            "*** Add File:",
        )):
            s = self.read_line()
            if not s.startswith("+"):
                raise DiffError(f"Invalid Add File line (missing '+'): {s}")
            lines.append(s[1:])
        return PatchAction(type=ActionType.ADD, new_file="\n".join(lines))

    def lines_for_context(self, lines: List[str]) -> List[str]:
        # helper to feed canonical when needed
        return lines


# --------------------------------------------------------------------------- #
#  Layered context-finders
# --------------------------------------------------------------------------- #
def find_context_core(
    lines: List[str], context: List[str], start: int
) -> Tuple[int, int]:
    n = len(context)
    norm_ctx = [canonical(x) for x in context]
    L = len(lines)

    # (a) exact on raw
    for i in range(start, L - n + 1):
        if lines[i : i + n] == context:
            return i, 0

    # (b) exact on canonical
    for i in range(start, L - n + 1):
        if [canonical(x) for x in lines[i : i + n]] == norm_ctx:
            return i, 1_000

    # (c) stripped match
    for i in range(start, L - n + 1):
        if [x.strip() for x in lines[i : i + n]] == [x.strip() for x in context]:
            return i, 10_000

    # (d) fuzzy first-line on canonical
    target = norm_ctx[0]
    best_i, best_r = -1, 0.0
    for i, ln in enumerate(lines[start:], start):
        r = difflib.SequenceMatcher(None, canonical(ln), target).ratio()
        if r > best_r:
            best_r, best_i = r, i
    if best_r > 0.75:
        return best_i, 50_000

    return -1, 0


def find_context(
    lines: List[str], context: List[str], start: int, eof: bool
) -> Tuple[int, int]:
    if eof:
        idx, fuzz = find_context_core(lines, context, len(lines) - len(context))
        if idx != -1:
            return idx, fuzz
        idx, fuzz2 = find_context_core(lines, context, start)
        return idx, fuzz2 + 10_000
    return find_context_core(lines, context, start)


def peek_next_section(
    lines: List[str], index: int
) -> Tuple[List[str], List[Chunk], int, bool]:
    old: List[str] = []
    del_lines: List[str] = []
    ins_lines: List[str] = []
    chunks: List[Chunk] = []
    orig_index = index
    mode = "keep"

    while index < len(lines):
        s = lines[index]
        if s.startswith((
            "@@",
            "*** End Patch",
            "*** Update File:",
            "*** Delete File:",
            "*** Add File:",
            "*** End of File",
        )) or s == "***":
            break
        if s.startswith("***"):
            raise DiffError(f"Invalid Line: {s}")
        index += 1
        last = mode
        if s == "":
            s = " "
        c0 = s[0]
        if c0 == "+": mode = "add"
        elif c0 == "-": mode = "delete"
        elif c0 == " ": mode = "keep"
        else:
            raise DiffError(f"Invalid Line: {s}")
        line = s[1:]
        if mode == "keep" and last != mode:
            if ins_lines or del_lines:
                chunks.append(Chunk(
                    orig_index=len(old) - len(del_lines),
                    del_lines=del_lines[:],
                    ins_lines=ins_lines[:],
                ))
            del_lines, ins_lines = [], []
        if mode == "delete":
            del_lines.append(line); old.append(line)
        elif mode == "add":
            ins_lines.append(line)
        else:
            old.append(line)

    if ins_lines or del_lines:
        chunks.append(Chunk(orig_index=len(old) - len(del_lines), del_lines=del_lines, ins_lines=ins_lines))

    if index < len(lines) and lines[index] == "*** End of File":
        index += 1
        return old, chunks, index, True
    return old, chunks, index, False

# --------------------------------------------------------------------------- #
#  Patch → Commit and Commit application
# --------------------------------------------------------------------------- #
def _get_updated_file(text: str, action: PatchAction, path: str) -> str:
    if action.type is not ActionType.UPDATE:
        raise DiffError("_get_updated_file called with non-update action")
    orig_lines = text.split("\n")
    dest_lines: List[str] = []
    orig_index = 0

    for chunk in action.chunks:
        if chunk.orig_index > len(orig_lines):
            raise DiffError(
                f"{path}: chunk.orig_index {chunk.orig_index} exceeds file length"
            )
        if orig_index > chunk.orig_index:
            raise DiffError(
                f"{path}: overlapping chunks at {orig_index} > {chunk.orig_index}"
            )

        dest_lines.extend(orig_lines[orig_index : chunk.orig_index])
        orig_index = chunk.orig_index

        dest_lines.extend(chunk.ins_lines)
        orig_index += len(chunk.del_lines)

    dest_lines.extend(orig_lines[orig_index:])
    return "\n".join(dest_lines)


def patch_to_commit(patch: Patch, orig: Dict[str, str]) -> Commit:
    commit = Commit()
    for path, action in patch.actions.items():
        if action.type is ActionType.DELETE:
            commit.changes[path] = FileChange(
                type=ActionType.DELETE, old_content=orig[path]
            )
        elif action.type is ActionType.ADD:
            if action.new_file is None:
                raise DiffError("ADD action without file content")
            commit.changes[path] = FileChange(
                type=ActionType.ADD, new_content=action.new_file
            )
        elif action.type is ActionType.UPDATE:
            new_content = _get_updated_file(orig[path], action, path)
            commit.changes[path] = FileChange(
                type=ActionType.UPDATE,
                old_content=orig[path],
                new_content=new_content,
                move_path=action.move_path,
            )
    return commit


# --------------------------------------------------------------------------- #
#  User-facing helpers
# --------------------------------------------------------------------------- #
def text_to_patch(text: str, orig: Dict[str, str]) -> Tuple[Patch, int]:
    lines = text.splitlines()  # preserves blank lines, no strip()
    if (
        len(lines) < 2
        or not Parser._norm(lines[0]).startswith("*** Begin Patch")
        or Parser._norm(lines[-1]) != "*** End Patch"
    ):
        raise DiffError("Invalid patch text - missing sentinels")

    parser = Parser(current_files=orig, lines=lines, index=1)
    parser.parse()
    return parser.patch, parser.fuzz


def identify_files_needed(text: str) -> List[str]:
    lines = text.splitlines()
    return [
        line[len("*** Update File: ") :]
        for line in lines
        if line.startswith("*** Update File: ")
    ] + [
        line[len("*** Delete File: ") :]
        for line in lines
        if line.startswith("*** Delete File: ")
    ]


def identify_files_added(text: str) -> List[str]:
    lines = text.splitlines()
    return [
        line[len("*** Add File: ") :]
        for line in lines
        if line.startswith("*** Add File: ")
    ]


# --------------------------------------------------------------------------- #
#  File-system helpers
# --------------------------------------------------------------------------- #
def load_files(paths: List[str], open_fn: Callable[[str], str]) -> Dict[str, str]:
    return {path: open_fn(path) for path in paths}


def apply_commit(
    commit: Commit,
    write_fn: Callable[[str, str], None],
    remove_fn: Callable[[str], None],
) -> None:
    for path, change in commit.changes.items():
        if change.type is ActionType.DELETE:
            remove_fn(path)
        elif change.type is ActionType.ADD:
            if change.new_content is None:
                raise DiffError(f"ADD change for {path} has no content")
            write_fn(path, change.new_content)
        elif change.type is ActionType.UPDATE:
            if change.new_content is None:
                raise DiffError(f"UPDATE change for {path} has no new content")
            target = change.move_path or path
            write_fn(target, change.new_content)
            if change.move_path:
                remove_fn(path)


def process_patch(
    text: str,
    open_fn: Callable[[str], str],
    write_fn: Callable[[str, str], None],
    remove_fn: Callable[[str], None],
) -> str:
    if not text.startswith("*** Begin Patch"):
        raise DiffError("Patch text must start with *** Begin Patch")
    paths = identify_files_needed(text)
    orig = load_files(paths, open_fn)
    patch, _fuzz = text_to_patch(text, orig)
    commit = patch_to_commit(patch, orig)
    apply_commit(commit, write_fn, remove_fn)
    return "Done!"


# --------------------------------------------------------------------------- #
#  Default FS helpers
# --------------------------------------------------------------------------- #
def open_file(path: str) -> str:
    with open(path, "rt", encoding="utf-8") as fh:
        return fh.read()


def write_file(path: str, content: str) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wt", encoding="utf-8") as fh:
        fh.write(content)


def remove_file(path: str) -> None:
    pathlib.Path(path).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
#  CLI entry-point
# --------------------------------------------------------------------------- #
def main() -> None:
    import sys

    patch_text = sys.stdin.read()
    if not patch_text:
        print("Please pass patch text through stdin", file=sys.stderr)
        return
    try:
        result = process_patch(patch_text, open_file, write_file, remove_file)
    except DiffError as exc:
        print(exc, file=sys.stderr)
        return
    print(result)


if __name__ == "__main__":
    main()
