"""Deterministic state builder for filesystem."""

from __future__ import annotations

import copy
import random
from typing import Any

from src.live_mcp.state_seeders.common import (
    _reference_datetime,
)
def _filesystem_state(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    reference_date = _reference_datetime(seed).date()

    # Tree variant 1: home-office (original)
    tree_home_office: dict[str, dict[str, Any]] = {
        "/": {"type": "dir", "content": "", "permissions": "755", "owner": "root"},
        "/home": {"type": "dir", "content": "", "permissions": "755", "owner": "root"},
        "/home/user": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/notes.txt": {"type": "file",
            "content": "TODO: review design doc\nTODO: update tests\nDONE: fix login bug",
            "permissions": "644", "owner": "user"},
        "/home/user/script.sh": {"type": "file",
            "content": "#!/bin/bash\necho hello", "permissions": "755", "owner": "user"},
        "/home/user/projects": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/projects/README.md": {"type": "file",
            "content": "# Projects\nWork in progress.", "permissions": "644", "owner": "user"},
        "/home/user/projects/config.ini": {"type": "file",
            "content": "[server]\nhost=localhost\nport=8080\n[database]\nname=proddb",
            "permissions": "644", "owner": "user"},
        "/home/user/logs": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/logs/app.log": {"type": "file",
            "content": "2026-06-20 INFO Starting application\n2026-06-21 WARN Connection timeout\n2026-06-22 ERROR Database unreachable",
            "permissions": "644", "owner": "user"},
        "/home/user/logs/error.log": {"type": "file",
            "content": "2026-06-22 ERROR: null pointer in module auth\n2026-06-22 ERROR: stack overflow in parser",
            "permissions": "644", "owner": "user"},
        "/home/user/data": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/data/users.csv": {"type": "file",
            "content": "id,name,role\n1,Alice,admin\n2,Bob,user\n3,Charlie,user",
            "permissions": "644", "owner": "user"},
        "/home/user/data/report_2026.json": {"type": "file",
            "content": '{"revenue": 150000, "costs": 90000, "profit": 60000}',
            "permissions": "644", "owner": "user"},
        "/home/user/pipeline.sh": {"type": "file",
            "content": "#!/bin/bash\n# Build and deploy pipeline\nmake build\nmake test\nmake deploy",
            "permissions": "755", "owner": "user"},
        "/protected": {"type": "dir", "content": "", "permissions": "700", "owner": "root"},
        "/protected/config.secret": {"type": "file",
            "content": "secret_key=abc123\ndb_password=xyz789",
            "permissions": "600", "owner": "root"},
        "/protected/certs": {"type": "dir", "content": "", "permissions": "700", "owner": "root"},
        "/protected/certs/server.crt": {"type": "file",
            "content": "-----BEGIN CERTIFICATE-----\nMIID...\n-----END CERTIFICATE-----",
            "permissions": "600", "owner": "root"},
    }

    # Tree variant 2: dev-workspace
    tree_dev_workspace: dict[str, dict[str, Any]] = {
        "/": {"type": "dir", "content": "", "permissions": "755", "owner": "root"},
        "/home": {"type": "dir", "content": "", "permissions": "755", "owner": "root"},
        "/home/user": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/src": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/src/main.py": {"type": "file",
            "content": "import utils\n\ndef main():\n    config = utils.load_config()\n    print(f\"Starting {config['app_name']}\")\n\nif __name__ == '__main__':\n    main()",
            "permissions": "644", "owner": "user"},
        "/home/user/src/utils.py": {"type": "file",
            "content": "import json\n\ndef load_config():\n    with open('../config/app.json') as f:\n        return json.load(f)",
            "permissions": "644", "owner": "user"},
        "/home/user/config": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/config/app.json": {"type": "file",
            "content": '{"app_name": "MyApp", "version": "2.1.0", "debug": false}',
            "permissions": "644", "owner": "user"},
        "/home/user/config/secrets.env": {"type": "file",
            "content": "API_KEY=sk-1234567890abcdef\nDB_URL=postgresql://localhost:5432/mydb",
            "permissions": "600", "owner": "user"},
        "/home/user/tests": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/tests/test_main.py": {"type": "file",
            "content": "import pytest\nfrom src.main import main\n\ndef test_main_runs():\n    # smoke test\n    pass",
            "permissions": "644", "owner": "user"},
        "/home/user/tests/test_utils.py": {"type": "file",
            "content": "import unittest\n\nclass TestUtils(unittest.TestCase):\n    def test_config(self):\n        pass",
            "permissions": "644", "owner": "user"},
        "/home/user/docs": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/docs/api.md": {"type": "file",
            "content": "# API Reference\n\n## GET /health\nReturns 200 OK.\n\n## POST /users\nCreate a new user.",
            "permissions": "644", "owner": "user"},
        "/home/user/docs/changelog.txt": {"type": "file",
            "content": "v2.1.0: Added health endpoint\nv2.0.0: Breaking changes to auth\nv1.5.0: Initial release",
            "permissions": "644", "owner": "user"},
        "/home/user/Makefile": {"type": "file",
            "content": ".PHONY: all test clean\n\nall: build\n\nbuild:\n\tpip install -r requirements.txt\n\ntest:\n\tpytest tests/\n\nclean:\n\tfind . -name '__pycache__' -exec rm -rf {} +",
            "permissions": "644", "owner": "user"},
        "/protected": {"type": "dir", "content": "", "permissions": "700", "owner": "root"},
        "/protected/config.secret": {"type": "file",
            "content": "secret_key=abc123\ndb_password=xyz789",
            "permissions": "600", "owner": "root"},
        "/protected/certs": {"type": "dir", "content": "", "permissions": "700", "owner": "root"},
        "/protected/certs/server.crt": {"type": "file",
            "content": "-----BEGIN CERTIFICATE-----\nMIID...\n-----END CERTIFICATE-----",
            "permissions": "600", "owner": "root"},
    }

    # Tree variant 3: media-project
    tree_media_project: dict[str, dict[str, Any]] = {
        "/": {"type": "dir", "content": "", "permissions": "755", "owner": "root"},
        "/home": {"type": "dir", "content": "", "permissions": "755", "owner": "root"},
        "/home/user": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/media": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/media/images": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/media/images/photo.jpg": {"type": "file",
            "content": "JPEG_42KB_2304x1536_camera_2026",
            "permissions": "644", "owner": "user"},
        "/home/user/media/images/logo.png": {"type": "file",
            "content": "PNG_8KB_256x256_transparent",
            "permissions": "644", "owner": "user"},
        "/home/user/media/videos": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/media/videos/demo.mp4": {"type": "file",
            "content": "MP4_15MB_1920x1080_30fps_2min",
            "permissions": "644", "owner": "user"},
        "/home/user/scripts": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/scripts/process.sh": {"type": "file",
            "content": "#!/bin/bash\n# Batch resize images\nfor f in media/images/*.jpg; do\n    convert \"$f\" -resize 800x600 \"thumbnails/$(basename $f)\"\ndone",
            "permissions": "755", "owner": "user"},
        "/home/user/scripts/backup.sh": {"type": "file",
            "content": "#!/bin/bash\n# Backup media to archive\ntar -czf archive/media_backup_$(date +%Y%m%d).tar.gz media/",
            "permissions": "755", "owner": "user"},
        "/home/user/thumbnails": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/archive": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/archive/data_2025.tar.gz": {"type": "file",
            "content": "BINARY_GZIP_2MB_2025",
            "permissions": "644", "owner": "user"},
        "/home/user/archive/old_config.zip": {"type": "file",
            "content": "BINARY_ZIP_128KB_2024_config_files",
            "permissions": "644", "owner": "user"},
        "/home/user/README.md": {"type": "file",
            "content": "# Media Project\n\nOrganize and process media files.\n\n## Structure\n- media/ : source images and videos\n- scripts/ : processing scripts\n- archive/ : compressed backups",
            "permissions": "644", "owner": "user"},
        "/protected": {"type": "dir", "content": "", "permissions": "700", "owner": "root"},
        "/protected/config.secret": {"type": "file",
            "content": "secret_key=abc123\ndb_password=xyz789",
            "permissions": "600", "owner": "root"},
        "/protected/certs": {"type": "dir", "content": "", "permissions": "700", "owner": "root"},
        "/protected/certs/server.crt": {"type": "file",
            "content": "-----BEGIN CERTIFICATE-----\nMIID...\n-----END CERTIFICATE-----",
            "permissions": "600", "owner": "root"},
    }

    # Tree variant 4: web-project
    tree_web_project: dict[str, dict[str, Any]] = {
        "/": {"type": "dir", "content": "", "permissions": "755", "owner": "root"},
        "/home": {"type": "dir", "content": "", "permissions": "755", "owner": "root"},
        "/home/user": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/templates": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/templates/index.html": {"type": "file",
            "content": "<!DOCTYPE html>\n<html>\n<head>\n  <title>My App</title>\n  <link rel=\"stylesheet\" href=\"/static/style.css\">\n</head>\n<body>\n  <h1>Welcome</h1>\n  <script src=\"/static/app.js\"></script>\n</body>\n</html>",
            "permissions": "644", "owner": "user"},
        "/home/user/templates/style.css": {"type": "file",
            "content": "body { font-family: sans-serif; margin: 0; padding: 20px; }\nh1 { color: #333; }",
            "permissions": "644", "owner": "user"},
        "/home/user/static": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/static/app.js": {"type": "file",
            "content": "async function fetchData() {\n  const res = await fetch('/api/data');\n  const data = await res.json();\n  console.log(data);\n}\n\nfetchData();",
            "permissions": "644", "owner": "user"},
        "/home/user/static/favicon.ico": {"type": "file",
            "content": "ICO_16x16_binary",
            "permissions": "644", "owner": "user"},
        "/home/user/db": {"type": "dir", "content": "", "permissions": "700", "owner": "user"},
        "/home/user/db/schema.sql": {"type": "file",
            "content": "CREATE TABLE users (\n  id SERIAL PRIMARY KEY,\n  name TEXT NOT NULL,\n  email TEXT UNIQUE NOT NULL\n);\n\nCREATE TABLE orders (\n  id SERIAL PRIMARY KEY,\n  user_id INT REFERENCES users(id),\n  total DECIMAL(10,2)\n);",
            "permissions": "644", "owner": "user"},
        "/home/user/db/seed_data.json": {"type": "file",
            "content": '{"users": [{"name": "Admin", "email": "admin@test.com"}], "orders": []}',
            "permissions": "644", "owner": "user"},
        "/home/user/nginx.conf": {"type": "file",
            "content": "server {\n  listen 80;\n  server_name localhost;\n  root /home/user/templates;\n  location /api/ {\n    proxy_pass http://127.0.0.1:3000;\n  }\n}",
            "permissions": "644", "owner": "user"},
        "/home/user/package.json": {"type": "file",
            "content": '{\n  "name": "web-project",\n  "version": "1.0.0",\n  "scripts": {\n    "start": "node server.js",\n    "test": "jest"\n  },\n  "dependencies": {\n    "express": "^4.18.0"\n  }\n}',
            "permissions": "644", "owner": "user"},
        "/protected": {"type": "dir", "content": "", "permissions": "700", "owner": "root"},
        "/protected/config.secret": {"type": "file",
            "content": "secret_key=abc123\ndb_password=xyz789",
            "permissions": "600", "owner": "root"},
        "/protected/certs": {"type": "dir", "content": "", "permissions": "700", "owner": "root"},
        "/protected/certs/server.crt": {"type": "file",
            "content": "-----BEGIN CERTIFICATE-----\nMIID...\n-----END CERTIFICATE-----",
            "permissions": "600", "owner": "root"},
    }

    tree_templates = [tree_home_office, tree_dev_workspace, tree_media_project, tree_web_project]
    fs = {k: dict(v) for k, v in rng.choice(tree_templates).items()}
    session_path = f"/home/user/session_{seed}.txt"
    fs[session_path] = {
        "type": "file",
        "content": f"session={seed}\nnonce={rng.getrandbits(64):016x}",
        "permissions": rng.choice(["600", "640", "644"]),
        "owner": "user",
    }
    # Every baseline variant exposes at least one directly usable link and
    # one archive of each supported format.  Previously these entities only
    # existed in one randomly selected tree (and no tree contained a link),
    # so public readonly discovery could not ground readlink/extract tasks.
    archive_entry = {
        f"session_{seed}.txt": copy.deepcopy(fs[session_path]),
    }
    fs[f"/home/user/session_{seed}.link"] = {
        "type": "symlink",
        "target": session_path,
        "permissions": "777",
        "owner": "user",
    }
    fs[f"/home/user/session_{seed}.tar"] = {
        "type": "file",
        "content": "",
        "permissions": "644",
        "owner": "user",
        "archive_format": "tar",
        "archive_entries": copy.deepcopy(archive_entry),
    }
    fs[f"/home/user/session_{seed}.zip"] = {
        "type": "file",
        "content": "",
        "permissions": "644",
        "owner": "user",
        "archive_format": "zip",
        "archive_entries": copy.deepcopy(archive_entry),
    }
    return {"fs": fs, "cwd": "/home/user", "umask": "022",
            "current_date": reference_date.isoformat()}
