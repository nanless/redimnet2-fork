"""Read-only identity checks for the isolated v3 libsndfile runtime.

Importing this module never imports soundfile or changes the environment.
The caller must select LD_LIBRARY_PATH before starting its interpreter.
"""
import hashlib
import importlib
import os
from pathlib import Path
import stat


LIBRARY_BYTES = 3639184
LIBRARY_SHA256 = "c872fb84af2afc0e96b603e0000f0f407af86c49ad1126f2785e5722c7bb27d7"
SOUNDFILE_SHA256 = "c40329db956b2dd1aee349bea622f36630d3db7169752c43308f44c541603a57"
SOUNDFILE_VERSION = "0.12.1"
LIBSNDFILE_VERSION = "1.2.2"
MAPS_FILE = Path("/proc/self/maps")


class RuntimeRefused(RuntimeError):
    """The isolated decoder cannot be established from current evidence."""


def _need(condition, message):
    if not condition:
        raise RuntimeRefused(message)


def _identity(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_size,
            st.st_mtime_ns, st.st_ctime_ns)


def _real_file(raw):
    _need(isinstance(raw, (str, os.PathLike)) and bool(str(raw)),
          "Decoder file must have an absolute path")
    path = Path(raw)
    _need(path.is_absolute() and ".." not in path.parts,
          "Decoder file must have an exact absolute path")
    _need(path.resolve(strict=True) == path,
          "Decoder file or ancestor is a symlink/noncanonical path")
    _need(stat.S_ISREG(path.lstat().st_mode), "Decoder file is not a regular file")
    return path


def _checked_file(raw, expected_sha, size=None, mode=None):
    path = _real_file(raw)
    before = path.lstat()
    if mode is not None:
        _need(stat.S_IMODE(before.st_mode) == mode, "Decoder library must be mode444")
    if size is not None:
        _need(before.st_size == size, "Decoder library size differs")
    # A final-component link swap must not turn the checked read into a link read.
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        _need(_identity(opened) == _identity(before), "Decoder file changed before read")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        _need(_identity(os.fstat(fd)) == _identity(before), "Decoder file changed during read")
    finally:
        os.close(fd)
    _need(_identity(path.lstat()) == _identity(before) and path.resolve(strict=True) == path,
          "Decoder file changed after read")
    actual = digest.hexdigest()
    _need(actual == expected_sha, "Decoder file SHA256 differs: " + str(path))
    return {"path": str(path), "bytes": before.st_size, "sha256": actual,
            "mode": oct(stat.S_IMODE(before.st_mode)),
            "stat": list(_identity(before))}


def verify_runtime_files(policy):
    """Verify the exact Policy-selected library, using fixed content identity."""
    try:
        return _checked_file(policy.decoder_library, LIBRARY_SHA256,
                             LIBRARY_BYTES, 0o444)
    except RuntimeRefused:
        raise
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
        raise RuntimeRefused("Cannot verify decoder library: " + str(exc)) from exc


def _mapped_libraries(text):
    paths = {}
    for line in text.splitlines():
        if "libsndfile" not in line:
            continue
        fields = line.split(None, 5)
        _need(len(fields) == 6 and "libsndfile" in fields[5], "Malformed libsndfile mapping")
        raw = fields[5]
        _need(not raw.endswith(" (deleted)"), "Loaded libsndfile has been deleted/replaced")
        _need(raw.startswith("/") and "\\" not in raw,
              "Loaded libsndfile has an unsupported map path")
        major, minor = fields[3].split(":")
        identity = (int(major, 16), int(minor, 16), int(fields[4]))
        paths.setdefault(raw, set()).add(identity)
    return paths


def audit_loaded_decoder(policy):
    """Verify package bytes/version and the one libsndfile actually mapped.

    No dlopen override, package upgrade or environment mutation is attempted.
    A previously imported old library therefore fails closed.
    """
    try:
        library = verify_runtime_files(policy)
        sf = importlib.import_module("soundfile")
        _need(getattr(sf, "__version__", None) == SOUNDFILE_VERSION,
              "Wrong soundfile Python version")
        _need(getattr(sf, "__libsndfile_version__", None) == LIBSNDFILE_VERSION,
              "Wrong loaded libsndfile version")
        package = _checked_file(getattr(sf, "__file__", None), SOUNDFILE_SHA256)
        _need(Path(package["path"]).suffix == ".py", "soundfile must identify its Python source")
        mapped = _mapped_libraries(MAPS_FILE.read_text(encoding="utf-8"))
        _need(set(mapped) == {str(policy.decoder_library)},
              "Expected exactly the Policy-selected loaded libsndfile path")
        dev, ino = library["stat"][:2]
        _need(mapped[str(policy.decoder_library)] == {(os.major(dev), os.minor(dev), ino)},
              "Mapped libsndfile device/inode differs from the checked file")
        after = verify_runtime_files(policy)
        _need(after == library, "Decoder library changed during loaded-runtime audit")
        return {"library": after, "soundfile": package,
                "soundfile_version": sf.__version__,
                "libsndfile_version": sf.__libsndfile_version__,
                "mapped_libraries": sorted(mapped), "verified": True}
    except RuntimeRefused:
        raise
    except (OSError, ValueError, TypeError, AttributeError, ImportError, RuntimeError) as exc:
        raise RuntimeRefused("Cannot audit loaded decoder: " + str(exc)) from exc
