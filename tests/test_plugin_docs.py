"""Tests for plugin documentation validation with path-traversal protection."""
import pytest

from fwbg_sdk.docs import DocsValidationResult, validate_plugin_docs


def _write(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestValidatePluginDocs:
    """Tests for validate_plugin_docs()."""

    def test_no_docs_dir_is_valid(self, tmp_path):
        result = validate_plugin_docs(tmp_path / "nonexistent")
        assert result.valid
        assert result.violations == []
        assert result.files == []

    def test_valid_docs_with_readme(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "README.md", "# My Plugin\n\nSome docs.")
        result = validate_plugin_docs(docs)
        assert result.valid
        assert "README.md" in result.files

    def test_missing_readme_is_violation(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "other.md", "# Not a readme")
        result = validate_plugin_docs(docs)
        assert not result.valid
        assert any(v.reason == "missing_readme" for v in result.violations)

    def test_valid_internal_link(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "README.md", "[details](advanced.md)")
        _write(docs / "advanced.md", "# Advanced")
        result = validate_plugin_docs(docs)
        assert result.valid

    def test_valid_image_link(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "images" / "chart.png", "fake png")
        _write(docs / "README.md", "![chart](images/chart.png)")
        result = validate_plugin_docs(docs)
        assert result.valid

    def test_valid_anchor_link(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "README.md", "[section](#my-section)")
        result = validate_plugin_docs(docs)
        assert result.valid

    def test_valid_external_url(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "README.md", "[example](https://example.com)")
        result = validate_plugin_docs(docs)
        assert result.valid

    def test_valid_mailto_link(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "README.md", "[mail](mailto:test@example.com)")
        result = validate_plugin_docs(docs)
        assert result.valid

    def test_path_traversal_rejected(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "README.md", "[evil](../../../etc/passwd)")
        result = validate_plugin_docs(docs)
        assert not result.valid
        violations = [v for v in result.violations if v.reason == "path_traversal"]
        assert len(violations) == 1
        assert "../../../etc/passwd" in violations[0].link

    def test_dotdot_in_middle_rejected(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "README.md", "[evil](images/../../secret.txt)")
        result = validate_plugin_docs(docs)
        assert not result.valid
        assert any(v.reason == "path_traversal" for v in result.violations)

    def test_absolute_path_rejected(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "README.md", "[evil](/etc/passwd)")
        result = validate_plugin_docs(docs)
        assert not result.valid
        violations = [v for v in result.violations if v.reason == "absolute_path"]
        assert len(violations) == 1

    def test_file_url_rejected(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "README.md", "[evil](file:///etc/passwd)")
        result = validate_plugin_docs(docs)
        assert not result.valid
        violations = [v for v in result.violations if v.reason == "external_local"]
        assert len(violations) == 1

    def test_missing_referenced_file(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "README.md", "[broken](nonexistent.md)")
        result = validate_plugin_docs(docs)
        assert not result.valid
        violations = [v for v in result.violations if v.reason == "missing_file"]
        assert len(violations) == 1

    def test_nested_docs_valid_relative_link(self, tmp_path):
        """A file in docs/sub/ linking ../images/x.png should be valid if within docs/."""
        docs = tmp_path / "docs"
        _write(docs / "README.md", "# Root")
        _write(docs / "images" / "chart.png", "fake")
        _write(docs / "sub" / "page.md", "![chart](../images/chart.png)")
        result = validate_plugin_docs(docs)
        assert result.valid

    def test_nested_docs_traversal_out_of_docs(self, tmp_path):
        """A file in docs/sub/ linking ../../ should be rejected."""
        docs = tmp_path / "docs"
        _write(docs / "README.md", "# Root")
        _write(docs / "sub" / "page.md", "[evil](../../secret.txt)")
        result = validate_plugin_docs(docs)
        assert not result.valid
        assert any(v.reason == "path_traversal" for v in result.violations)

    def test_multiple_violations(self, tmp_path):
        docs = tmp_path / "docs"
        _write(
            docs / "README.md",
            "[a](../secret)\n[b](/etc/passwd)\n[c](file:///tmp/x)\n",
        )
        result = validate_plugin_docs(docs)
        assert not result.valid
        assert len(result.violations) == 3
        reasons = {v.reason for v in result.violations}
        assert reasons == {"path_traversal", "absolute_path", "external_local"}

    def test_link_with_anchor_validated(self, tmp_path):
        """Links like file.md#section should validate the file part."""
        docs = tmp_path / "docs"
        _write(docs / "README.md", "[link](other.md#section)")
        _write(docs / "other.md", "# Other")
        result = validate_plugin_docs(docs)
        assert result.valid

    def test_link_with_anchor_missing_file(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "README.md", "[link](missing.md#section)")
        result = validate_plugin_docs(docs)
        assert not result.valid
        assert any(v.reason == "missing_file" for v in result.violations)

    def test_files_list_complete(self, tmp_path):
        docs = tmp_path / "docs"
        _write(docs / "README.md", "# Root")
        _write(docs / "advanced.md", "# Advanced")
        _write(docs / "images" / "chart.png", "fake")
        result = validate_plugin_docs(docs)
        assert result.valid
        assert sorted(result.files) == sorted(
            ["README.md", "advanced.md", "images/chart.png"]
        )

    def test_line_numbers_correct(self, tmp_path):
        docs = tmp_path / "docs"
        _write(
            docs / "README.md",
            "line 1\nline 2\n[bad](../evil)\nline 4\n",
        )
        result = validate_plugin_docs(docs)
        assert not result.valid
        assert result.violations[0].line == 3


class TestDocsValidationResult:
    """Tests for the result dataclass."""

    def test_empty_result_is_valid(self):
        result = DocsValidationResult(valid=True)
        assert result.valid
        assert result.violations == []
        assert result.files == []
