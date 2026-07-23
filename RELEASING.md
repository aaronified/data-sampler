# Releasing data-sampler

Releases are gated and manual-by-intent: nothing is uploaded to PyPI unless a
human publishes a GitHub Release (or manually dispatches the workflow). The
workflow uses **PyPI Trusted Publishing** (OIDC), so no API token is stored in
the repo, in CI secrets, or anywhere else.

## One-time setup (PyPI side)

1. Create a PyPI account at <https://pypi.org/account/register/> and enable 2FA.
2. Because `data-sampler` has no release on PyPI yet, register a **pending
   publisher**: <https://pypi.org/manage/account/publishing/> → "Add a new
   pending publisher" with:
   - PyPI project name: `data-sampler`
   - Owner: `aaronified`
   - Repository: `data-sampler`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. (Recommended) In GitHub: Settings → Environments → create `pypi` and add
   yourself as a required reviewer. The publish job then waits for your
   click-to-approve on every release.

After the first successful publish, the pending publisher becomes the project's
trusted publisher automatically.

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
