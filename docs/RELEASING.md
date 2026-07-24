# Releasing data-sampler

Releases are gated and manual-by-intent: nothing is uploaded to PyPI unless a
human publishes a GitHub Release (or manually dispatches the workflow). The
workflow uses **PyPI Trusted Publishing** (OIDC), so no API token is stored in
the repo, in CI secrets, or anywhere else.

## One-time setup (PyPI side) — DONE (v3.2.1, 2026-07-23)

Completed: the PyPI account exists, and the pending publisher (project
`data-sampler`, owner `aaronified`, repo `data-sampler`, workflow
`release.yml`, environment `pypi`) became the project's permanent **trusted
publisher** with the first publish. Nothing to repeat per release.

If it ever needs re-registering (new repo/owner/workflow name), manage it at
<https://pypi.org/manage/project/data-sampler/settings/publishing/>.

Still recommended: in GitHub → Settings → Environments → `pypi`, add yourself
as a required reviewer so every publish waits for your click-to-approve
(currently publishes run unattended once tests pass).

## Per release

1. Ensure `main` is green: `venv\Scripts\python -m pytest -q`.
2. Bump the version in **both** places (they must match):
   - `src/data_sampler/__init__.py` → `__version__`
   - `pyproject.toml` → `[project] version`
3. In `CHANGELOG.md`, retitle the `unreleased` section to the version + date.
4. Commit, tag, and push:

   ```sh
   git commit -am "Release vX.Y.Z"
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```

5. On GitHub: Releases → "Draft a new release" → choose the tag → paste the
   changelog section as the notes → **Publish release**. This triggers
   `.github/workflows/release.yml`: tests (Linux + Windows) → build →
   `twine check` → wheel smoke test → publish to PyPI (after environment
   approval, if configured).

## Manual fallback (no GitHub Actions)

```powershell
venv\Scripts\python -m pytest -q
venv\Scripts\python -m build
venv\Scripts\python -m twine check dist/*
venv\Scripts\python -m twine upload dist/*   # uses .pypirc or keyring; never paste tokens into chat/CI logs
```
