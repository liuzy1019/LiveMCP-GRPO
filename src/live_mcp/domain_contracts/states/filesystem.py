"""Filesystem state facts audited against its handler implementation."""

from src.live_mcp.domain_contracts.states.common import arg, facts, out


def _exists(name: str = "path"):
    return arg("file", name, "filesystem.exists")


def _type(name: str, value: str):
    return arg("file", name, "filesystem.type", value)


def _target(name: str, slot: str):
    return arg("file", name, slot, observed=False)


_TEXT_FILE_TOOLS = {
    "cat", "head", "tail", "wc", "sort", "uniq", "cut", "sed", "awk",
    "truncate", "split",
}

_BYTE_SAFE_FILE_TOOLS = {
    "md5sum", "sha256sum", "file_info", "xxd",
}


FILESYSTEM_STATE_FACTS = {
    "ls": facts(),
    "cd": facts(pre=(_exists(), _type("path", "dir"))),
    "pwd": facts(),
    "stat": facts(pre=(_exists(),)),
    "find": facts(),
    "grep": facts(),
    "tree": facts(),
    "mkdir": facts(
        pre=(_target("path", "filesystem.target_creatable"),),
        post=(
            out("file", "path", "filesystem.exists"),
            out("file", "path", "filesystem.type", "dir"),
        ),
    ),
    "touch": facts(
        pre=(_target("path", "filesystem.target_writable"),),
        post=(
            out("file", "path", "filesystem.exists"),
            out("file", "path", "filesystem.type", "file"),
            out("file", "path", "filesystem.archive", False),
        ),
    ),
    "mv": facts(
        pre=(
            _exists("source"),
            arg("file", "source", "filesystem.protected", False),
            _target("target", "filesystem.target_creatable"),
        ),
        post=(
            arg("file", "source", "filesystem.exists", False),
            out("file", "target", "filesystem.exists"),
        ),
    ),
    "cp": facts(
        pre=(
            _exists("source"),
            arg("file", "source", "filesystem.protected", False),
            _target("target", "filesystem.target_creatable"),
        ),
        post=(out("file", "target", "filesystem.exists"),),
    ),
    "rm": facts(
        pre=(
            _exists(),
            arg("file", "path", "filesystem.deletable"),
        ),
        post=(arg("file", "path", "filesystem.exists", False),),
    ),
    "chmod": facts(pre=(
        _exists(), arg("file", "path", "filesystem.protected", False),
    )),
    "chown": facts(pre=(
        _exists(), arg("file", "path", "filesystem.ownership_change_allowed"),
    )),
    "umask": facts(),
    "du": facts(),
    "df": facts(),
    "symlink": facts(
        pre=(_target("link_path", "filesystem.target_creatable"),),
        post=(
            out("file", "link_path", "filesystem.exists"),
            out("file", "link_path", "filesystem.type", "symlink"),
        ),
    ),
    "readlink": facts(pre=(_exists(), _type("path", "symlink"))),
    "tar_create": facts(
        pre=(_target("archive", "filesystem.target_creatable"),),
        post=(
            out("file", "archive", "filesystem.exists"),
            out("file", "archive", "filesystem.type", "file"),
            out("file", "archive", "filesystem.archive", True),
            out("file", "archive", "filesystem.archive_format", "tar"),
        ),
    ),
    "tar_extract": facts(pre=(
        arg("file", "archive", "filesystem.exists"),
        arg("file", "archive", "filesystem.archive", True),
        arg("file", "archive", "filesystem.archive_format", "tar"),
    )),
    "zip": facts(
        pre=(_target("archive", "filesystem.target_creatable"),),
        post=(
            out("file", "archive", "filesystem.exists"),
            out("file", "archive", "filesystem.type", "file"),
            out("file", "archive", "filesystem.archive", True),
            out("file", "archive", "filesystem.archive_format", "zip"),
        ),
    ),
    "unzip": facts(pre=(
        arg("file", "archive", "filesystem.exists"),
        arg("file", "archive", "filesystem.archive", True),
        arg("file", "archive", "filesystem.archive_format", "zip"),
    )),
    "diff": facts(pre=(
        _exists("file1"), _type("file1", "file"),
        arg("file", "file1", "filesystem.archive", False),
        _exists("file2"), _type("file2", "file"),
        arg("file", "file2", "filesystem.archive", False),
    )),
    "join": facts(pre=(
        _exists("file1"), _type("file1", "file"),
        arg("file", "file1", "filesystem.archive", False),
        _exists("file2"), _type("file2", "file"),
        arg("file", "file2", "filesystem.archive", False),
    )),
}

FILESYSTEM_STATE_FACTS.update({
    name: facts(pre=(
        _exists(),
        _type("path", "file"),
        arg("file", "path", "filesystem.archive", False),
    ))
    for name in _TEXT_FILE_TOOLS
})

FILESYSTEM_STATE_FACTS.update({
    name: facts(pre=(_exists(), _type("path", "file")))
    for name in _BYTE_SAFE_FILE_TOOLS
})
