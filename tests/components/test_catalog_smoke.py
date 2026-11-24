"""The checks that make the components tier and the catalog smoke able to fail.

Each runner is exercised on a planted defect as well as a clean case: a check whose failure path is never taken cannot
be told apart from one that does not run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import check_catalog, run_component_tests
from scripts.catalog_smoke import smoke_form

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG = REPO_ROOT / "catalog.yaml"


def _component(root: Path, name: str, value: str, passes: bool = True) -> None:
    """A component shipping an ``exp`` package (the name two real components share) and a test reading it."""
    (root / "components" / name / "exp").mkdir(parents=True)
    (root / "components" / name / "exp" / "__init__.py").write_text(f"VALUE = {value!r}\n")
    (root / "components" / name / "tests").mkdir()
    expected = value if passes else "something else"
    (root / "components" / name / "tests" / f"test_{name}.py").write_text(
        f"from exp import VALUE\n\n\ndef test_value():\n    assert VALUE == {expected!r}\n"
    )


class TestComponentTests:
    def test_components_sharing_a_package_name_each_import_their_own(self, tmp_path):
        _component(tmp_path, "one", "first")
        _component(tmp_path, "two", "second")
        assert run_component_tests.main(["--root", str(tmp_path)]) == 0

    def test_a_failing_component_fails_the_run_by_name(self, tmp_path, capsys):
        _component(tmp_path, "good", "first")
        _component(tmp_path, "bad", "second", passes=False)
        assert run_component_tests.main(["--root", str(tmp_path)]) == 1
        out = capsys.readouterr().out
        assert "FAILED: bad" in out and "passed: good" in out

    def test_a_named_component_without_tests_is_refused(self, tmp_path):
        _component(tmp_path, "one", "first")
        assert run_component_tests.main(["--root", str(tmp_path), "absent"]) == 2

    def test_the_published_layout_fails_a_test_that_resolves_out_of_its_snapshot(self, tmp_path, capsys):
        """The defect the layout exists to catch: ``resolve()`` follows a snapshot file into the blob store."""
        _component(tmp_path, "sibling", "first")
        tests = tmp_path / "components" / "sibling" / "tests"
        (tests / "data.txt").write_text("beside the test\n")
        (tests / "test_reads_a_sibling.py").write_text(
            "from pathlib import Path\n\n\ndef test_sibling():\n"
            "    assert (Path(__file__).resolve().parent / 'data.txt').is_file()\n"
        )
        assert run_component_tests.main(["--root", str(tmp_path), "sibling"]) == 0
        assert run_component_tests.main(["--root", str(tmp_path), "--published-layout", "sibling"]) == 1
        assert "FAILED: sibling" in capsys.readouterr().out

    def test_the_published_layout_empties_every_cache(self, tmp_path):
        env = run_component_tests.empty_caches(tmp_path)
        assert {"IT_CACHE", "IT_COMPONENTS_HUB_CACHE", "IT_ANALYSIS_HUB_CACHE", "HF_HUB_CACHE"} <= set(env)
        assert all(not Path(v).exists() for k, v in env.items() if k != "IT_TRUST_REMOTE_CODE")

    def test_this_repository_has_component_tests(self):
        """The tier runs what exists: an empty selection would pass while running nothing."""
        assert {"jlens_steering_ops", "concept-direction"} <= set(run_component_tests.component_test_dirs(REPO_ROOT))


class TestCatalogSmokeForms:
    @pytest.mark.parametrize(
        ("smoke", "form"), [("rte_demo.gpt2.core", "config"), ("tests/", "tests"), ("tests/cpu/", "tests")]
    )
    def test_form(self, smoke, form):
        assert smoke_form(smoke) == form

    def test_every_entry_with_a_smoke_is_first_party_and_checked_against_its_source(self):
        entries = yaml.safe_load(CATALOG.read_text())["components"]
        smoked = [e for e in entries if e["smoke"] is not None]
        assert smoked, "no catalog entry has a smoke, so the scheduled run would execute nothing"
        assert check_catalog.check(CATALOG) == []


class TestCatalogCheckRefusesAnUnsatisfiableSmoke:
    def _catalog(self, tmp_path: Path, smoke: str) -> Path:
        source = tmp_path / "components" / "thing"
        (source / "tests").mkdir(parents=True)
        (source / "it_component.yaml").write_text(
            yaml.safe_dump({"kinds": ["module"], "module": {"configs": {"thing.gpt2.core": "configs/x.yaml"}}})
        )
        entry = {
            "repo": "someone/thing",
            "revision": "0" * 40,
            "kinds": ["module"],
            "interpretune": ">=0.1.dev0",
            "owner": "someone",
            "smoke": smoke,
            "first_party": True,
        }
        path = tmp_path / "catalog.yaml"
        path.write_text(yaml.safe_dump({"components": [entry]}))
        return path

    def test_a_satisfiable_smoke_passes(self, tmp_path):
        assert check_catalog.check(self._catalog(tmp_path, "thing.gpt2.core")) == []
        assert check_catalog.check(self._catalog(tmp_path / "b", "tests/")) == []

    def test_an_unknown_configuration_is_refused_by_name(self, tmp_path):
        (problem,) = check_catalog.check(self._catalog(tmp_path, "thing.gpt2.typo"))
        assert "'thing.gpt2.typo' is not a module configuration" in problem and "thing.gpt2.core" in problem

    def test_a_missing_test_directory_is_refused_by_name(self, tmp_path):
        (problem,) = check_catalog.check(self._catalog(tmp_path, "cpu_tests/"))
        assert "'cpu_tests/' names a test directory" in problem


@pytest.mark.hf_live
def test_an_entry_whose_smoke_cannot_load_fails_the_run(tmp_path):
    """The control the scheduled run depends on: a smoke naming a configuration the pinned revision lacks fails it."""
    rte = next(e for e in yaml.safe_load(CATALOG.read_text())["components"] if e["repo"] == "speediedan/rte")
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(yaml.safe_dump({"components": [dict(rte, smoke="rte_demo.no_such_model.core")]}))
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "catalog_smoke.py"), str(catalog)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "no module configuration 'rte_demo.no_such_model.core'" in proc.stderr, proc.stderr
    assert "speediedan/rte: FAILED" in proc.stdout, proc.stdout


@pytest.mark.hf_live
def test_a_test_directory_the_published_snapshot_lacks_fails_the_run(tmp_path):
    """The test-directory form downloads the pinned snapshot and refuses a directory it does not carry, by name."""
    entry = {
        "repo": "speediedan/jlens_steering_ops",
        "revision": "ebf92e02d14313bc676acde37a17ed1960c1c594",
        "kinds": ["ops"],
        "interpretune": ">=0.1.dev0",
        "owner": "speediedan",
        "smoke": "no_such_tests/",
        "first_party": True,
    }
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(yaml.safe_dump({"components": [entry]}))
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "catalog_smoke.py"), str(catalog)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "smoke names 'no_such_tests/', which the published snapshot does not have" in proc.stderr, proc.stderr
