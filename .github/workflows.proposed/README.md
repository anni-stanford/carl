# Proposed GitHub Actions workflows

These YAMLs live here so they can be reviewed in the scaffold PR. **On merge,
rename `.github/workflows.proposed/` to `.github/workflows/`** to activate:

```bash
git mv .github/workflows.proposed .github/workflows
git commit -m "ci: enable proposed workflows"
git push
```

The OAuth token used to open the scaffold PR did not have GitHub's `workflow`
scope, so the workflow files had to be staged outside `.github/workflows/`.
The maintainer's normal browser session has `workflow` scope and the rename
above takes one commit.
