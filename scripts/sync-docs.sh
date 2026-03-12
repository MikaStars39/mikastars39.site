#!/bin/bash
# Sync docs from a GitHub repository and regenerate manifest.json
# Edit the two variables below before use.

SITE_DIR="/path/to/mikastars39.site"          # TODO: set to your actual site root
REPO_URL="https://github.com/USER/REPO.git"   # TODO: set to your docs repo URL

DOCS_DIR="$SITE_DIR/docs"

# First run: clone; subsequent runs: pull
if [ -d "$DOCS_DIR/.git" ]; then
    git -C "$DOCS_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$DOCS_DIR"
fi

# Regenerate manifest.json listing all .md files
cd "$DOCS_DIR"
python3 - <<'EOF'
import os, json, re

files = []
for root, dirs, filenames in os.walk("."):
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for f in sorted(filenames):
        if f.endswith(".md"):
            rel = os.path.join(root, f).lstrip("./")
            title = re.sub(r'[-_]', ' ', f[:-3])
            files.append({"path": rel, "title": title})

with open("manifest.json", "w") as fp:
    json.dump(files, fp, indent=2, ensure_ascii=False)
print(f"Generated manifest.json with {len(files)} files")
EOF
