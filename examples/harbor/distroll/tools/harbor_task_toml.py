from __future__ import annotations

import re


def set_toml_value(text: str, section: str, key: str, value: str) -> str:
    lines = text.splitlines()
    header = f"[{section}]"
    try:
        start = lines.index(header)
    except ValueError:
        lines.extend(["", header, f"{key} = {value}"])
        return "\n".join(lines) + "\n"
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if re.fullmatch(r"\[.+]", lines[index].strip())
        ),
        len(lines),
    )
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for index in range(start + 1, end):
        if pattern.match(lines[index]):
            lines[index] = f"{key} = {value}"
            return "\n".join(lines) + "\n"
    lines.insert(end, f"{key} = {value}")
    return "\n".join(lines) + "\n"


def remove_toml_value(text: str, section: str, key: str) -> str:
    lines = text.splitlines()
    header = f"[{section}]"
    if header not in lines:
        return text
    start = lines.index(header)
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if re.fullmatch(r"\[.+]", lines[index].strip())
        ),
        len(lines),
    )
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    lines = [
        line
        for index, line in enumerate(lines)
        if not (start < index < end and pattern.match(line))
    ]
    return "\n".join(lines) + "\n"
