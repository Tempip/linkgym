# Releasing

Maintainer checklist for a linkgym release on PyPI and, when the data changes, a dataset
release on Zenodo.

## One-time setup

1. **PyPI Trusted Publisher** (pypi.org, project `linkgym`, Settings -> Publishing): owner
   `Tempip`, repository `linkgym`, workflow `publish.yml`, environment `pypi`. No API token
   or secret: the workflow's `id-token: write` permission lets PyPI verify it over OIDC.
2. **GitHub environment `pypi`** (repository Settings -> Environments). Under "Deployment
   branches and tags", choose "Selected branches and tags" and add a rule of type **Tag**
   with the pattern `v*`.

   > **Pitfall: the rule must be a tag rule, not a branch rule.** The publish job runs on
   > the release's tag, and a *branch* rule `v*` does not match tag refs. With a branch
   > rule, the 0.1.0 publish job failed with "Tag v0.1.0 is not allowed to deploy to pypi".
   > If that happens, change the rule to a tag rule and re-run the failed job.

3. **Zenodo GitHub integration** (zenodo.org, account menu -> GitHub): sync, then switch on
   `Tempip/linkgym`. Zenodo then archives every GitHub release published *afterwards* and
   gives it a DOI; releases published before the switch was on are not archived.
   - **Metadata.** Zenodo reads `CITATION.cff` (a subset of its fields: title, version,
     abstract, authors with affiliation and ORCID, keywords, license, type). If the
     repository also contains a `.zenodo.json`, Zenodo uses only that file and ignores
     `CITATION.cff` entirely. This repository has no `.zenodo.json`; keep `CITATION.cff`
     current instead.
   - **No pre-reserved DOI.** A DOI cannot be reserved before a release archived by the
     integration. The software DOI exists only once the GitHub release is published;
     add it to the README afterwards if wanted.

## Dataset release (only when the data changes)

The Munich v1 files are published as one Zenodo record: record 23135098, version DOI
`10.5281/zenodo.23135098`; concept DOI (all versions) `10.5281/zenodo.23135097`.

- Files of a published Zenodo version never change. A new dataset version is a new
  version record with its own version DOI, and a new name in `linkgym.datasets.DATASETS`
  (e.g. `munich-v2`). Never change the record, file name or SHA-256 behind an existing
  name.
- Create a new version with "New version" on the published record, not with a new
  upload: it then shares the concept DOI `10.5281/zenodo.23135097`. Steps 2-5 below apply
  to it, with a new record number.
- `fetch` points to the files of the version record. The concept DOI (all versions,
  resolves to the latest) is for "latest version" references only.

1. Build the package outside the repository and check it:

   ```bash
   python examples/rt/package_zenodo.py            # -> ../zenodo/munich-v1
   ```

2. On zenodo.org, create a new upload (fields in `examples/rt/zenodo/ZENODO_FORM.md`). In
   "Digital Object Identifier", answer "No" to "Do you already have a DOI for this
   upload?" and click **Get a DOI now!**: this reserves the record's version DOI,
   `10.5281/zenodo.<record>`. Save the draft; deleting the draft loses the reserved DOI.
3. Set `ZENODO_RECORD = "<record>"` in `src/linkgym/datasets.py`, replace the
   `10.5281/zenodo.XXXXXXX` placeholders in `README.md` (and wherever
   `grep -rn XXXXXXX README.md docs src` finds them), rebuild the package (step 1; its
   README now carries the DOI) and commit.
4. Upload all files of the package folder to the draft, check the preview, publish.
5. After publishing: note the concept DOI shown on the record page ("Cite all versions"),
   and check a real download from an empty cache:

   ```bash
   LINKGYM_DATA_DIR=$(mktemp -d) python -m linkgym.datasets munich-v1 munich-v1-test-alt
   ```

## Software release

1. **Version and texts.** `version` in `pyproject.toml`; in `CHANGELOG.md`, move
   `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD` and update the links at the bottom;
   `version` and `date-released` in `CITATION.cff`; the version in the README's BibTeX.
   The dates are those of the release-prep commit: if the tag is created on a later day,
   update them first. In `CITATION.cff`, remove the previous release's version DOI from
   `identifiers` (keep the concept DOI in `doi`): the new one exists only after the release.
   Keep `CITATION.cff` valid (`cffconvert --validate`): if Zenodo cannot parse it, it
   archives nothing.
2. **No placeholders left:** `grep -rn XXXXXXX README.md docs src CITATION.cff` finds
   nothing (the dataset record must be published first, see above).
3. **Checks:**

   ```bash
   ruff check . && ruff format --check .
   pytest -q                       # everything, including slow tests
   rm -rf dist && python -m build && twine check --strict dist/*
   ```

   In an environment with the `rt` extra and a CUDA GPU, also `pytest -q -m rt`. Install
   the wheel in a fresh environment and run
   `python -c "import linkgym; print(linkgym.__version__)"`.
4. **Push** `main` and wait for CI to pass.
5. **Zenodo integration on** for `Tempip/linkgym` (setup step 3) **before publishing the
   GitHub release**, or the release gets no software DOI.
6. **Tag** the release commit and push the tag (pushing a tag alone publishes nothing):

   ```bash
   git tag -a vX.Y.Z -m "linkgym X.Y.Z"
   git push origin vX.Y.Z
   ```

7. **GitHub release** from the tag (title `linkgym X.Y.Z`, notes from the CHANGELOG
   section), then **Publish release**. This triggers `publish.yml` (build, `twine check`,
   upload to PyPI through the `pypi` environment) and the Zenodo archive.
8. **Verify:** the PyPI page shows X.Y.Z and renders the README; `pip install
   linkgym==X.Y.Z` works in a fresh environment; the Zenodo software record exists with the
   metadata from `CITATION.cff`.
9. **Afterwards:** add the version DOI that Zenodo minted for the release to
   `identifiers` in `CITATION.cff` (the concept DOI of the software is
   `10.5281/zenodo.23135621`), and a new empty `[Unreleased]` section to the CHANGELOG if it
   is missing.
