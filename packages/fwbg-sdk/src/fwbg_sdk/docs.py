"""Plugin documentation validation with path-traversal protection.

Plugins may include a docs/ directory with Markdown files and images.
All links within documentation must resolve within the docs/ directory —
no path traversal, no absolute paths, no access outside the plugin's
documentation boundary.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Matches [text](path) and ![img](path), captures the path
_LINK_RE = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")

# Schemes that are allowed to pass without file validation
_EXTERNAL_SCHEMES = ("http://", "https://", "mailto:")


@dataclass
class DocsViolation:
    """A single documentation validation violation."""

    file: str
    line: int
    link: str
    reason: str  # "path_traversal" | "absolute_path" | "missing_file" | "missing_readme" | "external_local"


@dataclass
class DocsValidationResult:
    """Result of validating a plugin's documentation directory."""

    valid: bool
    violations: List[DocsViolation] = field(default_factory=list)
    files: List[str] = field(default_factory=list)


def validate_plugin_docs(docs_dir: Path) -> DocsValidationResult:
    """Validate plugin documentation for path safety.

    Checks that all links in Markdown files resolve within the docs/
    directory. Rejects path traversal, absolute paths, and file:// URLs.

    Args:
        docs_dir: Path to the plugin's docs/ directory.

    Returns:
        DocsValidationResult with violations (if any).
    """
    docs_dir = docs_dir.resolve()

    if not docs_dir.is_dir():
        return DocsValidationResult(valid=True)

    violations: List[DocsViolation] = []
    all_files: List[str] = []

    # Collect all files
    for f in docs_dir.rglob("*"):
        if f.is_file():
            all_files.append(str(f.relative_to(docs_dir)))

    # Check README.md exists
    if not (docs_dir / "README.md").exists():
        violations.append(
            DocsViolation(
                file="",
                line=0,
                link="README.md",
                reason="missing_readme",
            )
        )

    # Validate all markdown files
    for md_file in docs_dir.rglob("*.md"):
        rel_path = str(md_file.relative_to(docs_dir))
        _validate_markdown_file(md_file, docs_dir, rel_path, violations)

    return DocsValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        files=all_files,
    )


def _validate_markdown_file(
    md_file: Path,
    docs_dir: Path,
    rel_path: str,
    violations: List[DocsViolation],
) -> None:
    """Validate all links in a single markdown file."""
    try:
        content = md_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    for line_num, line in enumerate(content.splitlines(), start=1):
        for match in _LINK_RE.finditer(line):
            link_target = match.group(2).strip()
            _validate_link(link_target, md_file, docs_dir, rel_path, line_num, violations)


def _validate_link(
    link: str,
    md_file: Path,
    docs_dir: Path,
    rel_path: str,
    line_num: int,
    violations: List[DocsViolation],
) -> None:
    """Validate a single link target."""
    # Skip external URLs
    if any(link.startswith(scheme) for scheme in _EXTERNAL_SCHEMES):
        return

    # Skip anchor-only links
    if link.startswith("#"):
        return

    # Strip anchor from path (e.g., "file.md#section" -> "file.md")
    link_path = link.split("#")[0]
    if not link_path:
        return

    # Reject file:// URLs
    if link_path.startswith("file://"):
        violations.append(
            DocsViolation(file=rel_path, line=line_num, link=link, reason="external_local")
        )
        return

    # Reject absolute paths
    if link_path.startswith("/"):
        violations.append(
            DocsViolation(file=rel_path, line=line_num, link=link, reason="absolute_path")
        )
        return

    # Resolve relative to the markdown file's directory and check containment
    resolved = (md_file.parent / link_path).resolve()

    # Must stay within docs_dir (catches all path traversal including ../)
    try:
        resolved.relative_to(docs_dir)
    except ValueError:
        violations.append(
            DocsViolation(file=rel_path, line=line_num, link=link, reason="path_traversal")
        )
        return

    # Referenced file must exist
    if not resolved.exists():
        violations.append(
            DocsViolation(file=rel_path, line=line_num, link=link, reason="missing_file")
        )
