"""Sphinx configuration for the interpretune examples documentation.

Built on Read the Docs as a subproject of interpretune.org, served at ``interpretune.org/projects/examples/``. The theme
configuration follows interpretune's own ``docs/source/conf.py`` so the two read as one site, and intersphinx links
the tutorials to interpretune's API and usage pages.
"""

from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DOCS_DIR.parent

project = "interpretune examples"
author = "Daniel Dale"
copyright = "2023-2026, Daniel Dale"

extensions = [
    "sphinx.ext.intersphinx",
    # myst_nb supersedes myst_parser (it activates it internally) and adds .ipynb as a source suffix. Do NOT also list
    # myst_parser: registering both raises an extension conflict.
    "myst_nb",
    "sphinx_copybutton",
]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "linkify", "dollarmath"]
myst_heading_anchors = 3
# As in interpretune's docs: a `$` in prose opens a math span that closes at the next `$` on the line, so spaced and
# digit-adjacent delimiters are refused at the parser.
myst_dmath_allow_space = False
myst_dmath_allow_digits = False

# --- Notebooks ---------------------------------------------------------------------------------------------------------
# NEVER execute a notebook at build time. The tutorials need GPUs, gated weights, the git-pinned adapter stack and, for
# the local-dashboard paths, a live Neuronpedia webapp. Outputs are baked in ahead of time (`nbpress-render`), so the
# docs render whatever the executed copy carries and the build stays a pure-CPU one.
nb_execution_mode = "off"
nb_merge_streams = True

# The notebooks live outside the pages' tree, so they are STAGED into `docs/notebooks/` (gitignored) at build time.
# Two sources, in priority order:
#   1. `docs/notebook_artifacts/`: executed copies WITH outputs, rendered on a host that can run them. These are what a
#      reader should see.
#   2. `tutorials/publish/`: the stripped, runnable notebooks users clone. A tutorial with no executed copy yet still
#      gets a page, code-only.
_NOTEBOOK_SOURCES = (_DOCS_DIR / "notebook_artifacts", _REPO_ROOT / "tutorials" / "publish")
_NOTEBOOK_STAGE_DIR = _DOCS_DIR / "notebooks"


def _stage_notebooks() -> None:
    """Copy the tutorials (and their sidecar assets) into the Sphinx source tree.

    A later source never overwrites an earlier one, so an executed copy always wins over the stripped fallback for the
    same relative path. The docs-excluded cells (the commented-out installer) are stripped here, with nbpress's own
    rules, so the rule holds for both sources.
    """
    import shutil

    from nbpress import render

    if _NOTEBOOK_STAGE_DIR.exists():
        shutil.rmtree(_NOTEBOOK_STAGE_DIR)
    staged: set[Path] = set()
    for source_dir in _NOTEBOOK_SOURCES:
        if not source_dir.is_dir():
            continue
        for source_path in sorted(source_dir.rglob("*")):
            if source_path.is_dir() or source_path.name.startswith("."):
                continue
            relative_path = source_path.relative_to(source_dir)
            if relative_path in staged:
                continue
            target_path = _NOTEBOOK_STAGE_DIR / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.suffix == ".ipynb":
                notebook, _ = render.strip_docs_excluded_cells(render.load_notebook(source_path))
                render.save_notebook(notebook, target_path)
            else:
                shutil.copy2(source_path, target_path)
            staged.add(relative_path)


_stage_notebooks()

templates_path = ["_templates"]
# The executed copies are staged into `notebooks/`; the originals are not pages of their own.
exclude_patterns = ["_build", "notebook_artifacts", "requirements.txt"]

intersphinx_mapping = {
    "interpretune": ("https://interpretune.org/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

linkcheck_ignore = [
    r"https://colab\.research\.google\.com/.*",
    r"https?://(localhost|127\.0\.0\.1)[:/].*",
]

html_theme = "pydata_sphinx_theme"
html_title = "Interpretune examples"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/images/logos/logo_interpretune.svg"
html_favicon = "_static/images/logos/icon_interpretune.svg"
html_theme_options = {
    "github_url": "https://github.com/interpretune/examples",
    "navbar_align": "content",
    "show_toc_level": 2,
    "show_nav_level": 0,
    "navigation_with_keys": False,
    # The library's docs are the parent site; the header points back to them on every page.
    "external_links": [{"name": "Interpretune docs", "url": "https://interpretune.org/en/latest/"}],
}
html_sidebars = {"**": ["sidebar-nav-global.html"]}
html_context = {
    "github_user": "interpretune",
    "github_repo": "examples",
    "github_version": "main",
    "doc_path": "docs",
}

suppress_warnings = [
    # The tutorials are authored for readers running them, where heading level is a styling choice rather than an
    # outline. Skipping a level or opening at H2 is deliberate there.
    "myst.header",
]
