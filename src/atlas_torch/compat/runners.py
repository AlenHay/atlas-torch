"""Backend boundary of the BasicATLAS-compatible API: one call per Fortran program run.

The API (:mod:`atlas_torch.compat.atlas`) writes the same run directories and shell command files as BasicATLAS
and executes those command files with a small built-in interpreter (:func:`run_script`). File operations
(``cd``, ``ln``, ``mv``, ``cp``, ``rm``, ...) are done in Python; every program invocation becomes ::

    run_program(name, workdir, stdin_text, unit_files) -> stdout_text

Program interface (what a torch implementation must do)
-------------------------------------------------------
``name``        one of :data:`PROGRAMS` (the executable's basename without ``.exe``).
``workdir``     absolute path; the Fortran program's current directory. Never ``os.chdir`` (programs of a
                parallel DFSYNTHE run execute concurrently in different workdirs): resolve paths against it.
``stdin_text``  the program's standard input (control cards / model deck), ``str``; ``""`` when none.
``unit_files``  ``{unit: path}`` for the pre-connected Fortran units. When called from a command file these are
                all ``fort.N`` entries present in ``workdir`` at launch (links to data files, or intermediate
                files left by the previous program of the chain); a direct caller may pass other paths.
                Unit N without an entry means the program would open ``workdir/fort.N`` (absent → new file).
Outputs         Write every output unit N to ``workdir/fort.N`` (unformatted units with gfortran's sequential
                record framing, or whatever the downstream program / reader of that file expects, bit-compatible
                with the Fortran), plus the named files the Fortran writes in its cwd (``progress.dat`` for
                ``synthe``); named inputs (``c12c13.dat`` for ``rmolecasc``, ``mask.bin`` for ``spectrv``) are
                read from ``workdir``. Scratch units that the Fortran leaves behind should be left behind too
                (later programs of the chain read them, the command file deletes them).
Return value    everything the program writes to standard output (the command file redirects it to e.g.
                ``output_main.out``, ``synthe.out``), as ``str`` (bytes are mapped 1:1 via latin-1).
Errors          raise; the API surfaces a ``ValueError`` like BasicATLAS does for any stderr output.

Registering an implementation::

    from atlas_torch.compat.runners import register

    @register("spectrv")
    def spectrv(workdir, stdin_text, unit_files):
        ...
        return stdout_text

or define a module-level ``PROGRAMS = {"spectrv": fn, ...}`` in one of :data:`TORCH_MODULES` (or in a module
listed in ``$ATLAS_TORCH_RUNNER_MODULES``, comma-separated); those modules are imported on first use.

Backend selection (``$ATLAS_TORCH_BACKEND``, read at every call): ``auto`` (default) uses the torch
implementation when one is registered and the stock Fortran executable otherwise; ``torch`` raises
``NotImplementedError`` for programs without a torch implementation; ``fortran`` always runs the executables in
``$ATLAS_TORCH_FORTRAN_BIN`` (default: ``<BasicATLAS>/bin``) — a development/test fallback that is not part of
the torch port.
"""
import glob
import importlib
import os
import shlex
import shutil
import subprocess
import threading
import warnings

from atlas_torch.compat.settings import python_path

__all__ = ["PROGRAMS", "TORCH_MODULES", "register", "registered", "backend", "run_program", "FortranRunner",
           "run_script", "ScriptError"]

PROGRAMS = ("xnfdf", "dfsynthe", "dfsortp", "separatedf", "kappa9", "kapreadts", "atlas9mem", "xnfpelsyn", "synbeg",
            "rgfalllinesnew", "rmolecasc", "rh2ofast", "synthe", "spectrv")

# modules that provide torch programs (imported lazily; missing modules are skipped)
TORCH_MODULES = ["atlas_torch.atlas9.program", "atlas_torch.kappa9.kappa9_prog", "atlas_torch.compat.torch_programs"]

_registry = {}
_discovered = False
_lock = threading.RLock()


def register(name, fn=None):
    """Register ``fn(workdir, stdin_text, unit_files) -> stdout_text`` as the torch implementation of ``name``.
    Usable as a decorator (``@register("synthe")``)."""
    if name not in PROGRAMS:
        raise ValueError("unknown program {!r}; expected one of {}".format(name, PROGRAMS))
    if fn is None:
        return lambda f: register(name, f)
    _registry[name] = fn
    return fn


def _discover():
    global _discovered
    with _lock:
        if _discovered:
            return
        _discovered = True
        extra = [m for m in os.environ.get("ATLAS_TORCH_RUNNER_MODULES", "").split(",") if m.strip()]
        for mod in TORCH_MODULES + [m.strip() for m in extra]:
            try:
                module = importlib.import_module(mod)
            except ModuleNotFoundError as e:
                if e.name is not None and (mod == e.name or mod.startswith(e.name + ".")):
                    continue                     # module not written (yet)
                warnings.warn("torch runner module {} failed to import: {!r}".format(mod, e))
                continue
            except Exception as e:               # a half-written port must not break the Fortran fallback
                warnings.warn("torch runner module {} failed to import: {!r}".format(mod, e))
                continue
            for name, fn in getattr(module, "PROGRAMS", {}).items():
                register(name, fn)


def registered():
    """{name: torch implementation} currently available."""
    _discover()
    return dict(_registry)


def backend():
    b = os.environ.get("ATLAS_TORCH_BACKEND", "auto").strip().lower() or "auto"
    if b not in ("auto", "torch", "fortran"):
        raise ValueError("ATLAS_TORCH_BACKEND must be auto, torch or fortran, not {!r}".format(b))
    return b


class FortranRunner:
    """Runs the stock executable ``<bindir>/<name>.exe`` in ``workdir`` (dev/test fallback).

    ``unit_files`` entries that are not yet present as ``workdir/fort.N`` are symlinked for the duration of
    the run (entries already present must be the same file)."""

    def __init__(self, bindir=None):
        self.bindir = bindir or os.environ.get("ATLAS_TORCH_FORTRAN_BIN") or os.path.join(python_path, "bin")

    def __call__(self, name, workdir, stdin_text, unit_files):
        exe = os.path.join(self.bindir, name + ".exe")
        if not os.path.isfile(exe):
            raise FileNotFoundError("Fortran executable {} not found".format(exe))
        made = []
        try:
            for unit, path in sorted((unit_files or {}).items()):
                link = os.path.join(workdir, "fort.{}".format(int(unit)))
                if os.path.lexists(link):
                    if not (os.path.exists(path) and os.path.exists(link) and os.path.samefile(link, path)):
                        raise FileExistsError("{} exists and is not {}".format(link, path))
                    continue
                os.symlink(os.path.abspath(path), link)
                made.append(link)
            p = subprocess.run([exe], cwd=workdir, input=(stdin_text or "").encode("latin-1"),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        finally:
            for link in made:
                if os.path.islink(link):
                    os.unlink(link)
        out, err = p.stdout.decode("latin-1"), p.stderr.decode("latin-1").strip()
        if p.returncode != 0 or err:
            raise ValueError("{} returned an error (exit code {}): {}".format(exe, p.returncode, err))
        return out


_fortran = FortranRunner()


def run_program(name, workdir, stdin_text="", unit_files=None):
    """Run program ``name`` (see module docstring); returns its standard output."""
    if name not in PROGRAMS:
        raise ValueError("unknown program {!r}".format(name))
    workdir = os.path.abspath(workdir)
    unit_files = dict(unit_files or {})
    b = backend()
    if b != "fortran":
        fn = registered().get(name)
        if fn is not None:
            return fn(workdir, stdin_text or "", unit_files)
        if b == "torch":
            raise NotImplementedError("no torch implementation registered for {} (ATLAS_TORCH_BACKEND=torch)".format(name))
    return _fortran(name, workdir, stdin_text or "", unit_files)


# ------------------------------------------------------------------------------------------------------------
# command-file interpreter
# ------------------------------------------------------------------------------------------------------------
class ScriptError(ValueError):
    pass


def _units(cwd):
    units = {}
    for path in glob.glob(os.path.join(glob.escape(cwd), "fort.*")):
        suffix = os.path.basename(path)[5:]
        if suffix.isdigit() and os.path.exists(path):
            units[int(suffix)] = path
    return units


def _parse_program(line):
    """'<dir>/<name>.exe [<<EOF | <<"EOF" | < file] [> file]' -> (name, heredoc_tag|None, stdin_file, stdout_file)."""
    head, _, rest = line.partition(".exe")
    name = os.path.basename(head)
    rest = rest.strip()
    tag = stdin_file = stdout_file = None
    while rest:
        if rest.startswith("<<"):
            rest = rest[2:].lstrip()
            word, rest = _word(rest)
            tag = word
        elif rest.startswith("<"):
            stdin_file, rest = _word(rest[1:].lstrip())
        elif rest.startswith(">"):
            stdout_file, rest = _word(rest[1:].lstrip())
        else:
            raise ScriptError("cannot parse program line {!r}".format(line))
        rest = rest.strip()
    return name, tag, stdin_file, stdout_file


def _word(s):
    """Leading shell word of s (optionally double-quoted) up to whitespace or a redirection."""
    if s.startswith('"'):
        end = s.index('"', 1)
        return s[1:end], s[end + 1:]
    i = 0
    while i < len(s) and not s[i].isspace() and s[i] not in "<>":
        i += 1
    return s[:i], s[i:]


def run_script(path, program=None):
    """Execute a BasicATLAS command file (the bash subset its templates use) and return the standard output
    that is not redirected to files. Programs go through ``program`` (default :func:`run_program`).

    Like ``bash`` + BasicATLAS's ``cmd()``, failures of individual commands do not stop the script; the
    collected errors are raised as one :class:`ScriptError` at the end."""
    program = program or run_program
    with open(path) as f:
        lines = f.read().split("\n")
    cwd = os.getcwd()
    stdout, errors = [], []

    def at(p):
        return os.path.join(cwd, os.path.expanduser(p))

    def matches(pattern):
        hits = sorted(glob.glob(at(pattern))) if glob.has_magic(pattern) else [at(pattern)]
        return [h for h in hits if os.path.lexists(h)] or None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        try:
            if ".exe" in line.split()[0]:
                name, tag, stdin_file, stdout_file = _parse_program(line)
                text = ""
                if tag is not None:
                    quoted = lines[i - 1].split("<<", 1)[1].lstrip().startswith(('"', "'"))
                    body = []
                    while lines[i] != tag:
                        body.append(lines[i] + "\n")
                        i += 1
                    i += 1
                    text = "".join(body)
                    if not quoted and any(c in text for c in "$`\\"):
                        raise ScriptError("unquoted here-document with shell expansions is not supported")
                elif stdin_file is not None:
                    with open(at(stdin_file), encoding="latin-1", newline="") as f:
                        text = f.read()
                out = program(name, cwd, text, _units(cwd))
                if stdout_file is not None:
                    with open(at(stdout_file), "w", encoding="latin-1", newline="") as f:
                        f.write(out)
                else:
                    stdout.append(out)
                continue

            words = shlex.split(line)
            cmd, args = words[0], words[1:]
            if cmd == "cd":
                cwd = os.path.realpath(at(args[0]))
            elif cmd == "mkdir":
                os.makedirs(at(args[-1]), exist_ok="-p" in args)
            elif cmd == "ln":
                if args[0] == "-s":
                    os.symlink(args[1], at(args[2]))
                else:
                    os.link(at(args[0]), at(args[1]))
            elif cmd == "mv":
                os.replace(at(args[0]), at(args[1]))
            elif cmd == "cp":
                shutil.copyfile(at(args[0]), at(args[1]))
            elif cmd == "rm":
                flags = "".join(a[1:] for a in args if a.startswith("-"))
                for pattern in (a for a in args if not a.startswith("-")):
                    hits = matches(pattern)
                    if hits is None:
                        if "f" not in flags:
                            raise FileNotFoundError("rm: cannot remove '{}': No such file or directory".format(pattern))
                        continue
                    for h in hits:
                        if os.path.isdir(h) and not os.path.islink(h):
                            if "r" not in flags:
                                raise IsADirectoryError("rm: cannot remove '{}': Is a directory".format(pattern))
                            shutil.rmtree(h)
                        else:
                            os.unlink(h)
            elif cmd == "echo":
                # echo WORDS > FILE
                if ">" not in words:
                    stdout.append(" ".join(args) + "\n")
                else:
                    k = words.index(">")
                    with open(at(words[k + 1]), "w") as f:
                        f.write(" ".join(words[1:k]) + "\n")
            elif cmd == "cat" and line.startswith("cat <<"):
                # cat <<"TAG" >FILE
                rest = line[len("cat <<"):].lstrip()
                tag, rest = _word(rest)
                target, _ = _word(rest.strip()[1:].lstrip())
                body = []
                while lines[i] != tag:
                    body.append(lines[i] + "\n")
                    i += 1
                i += 1
                with open(at(target), "w") as f:
                    f.write("".join(body))
            else:
                raise ScriptError("unsupported command {!r}".format(line))
        except (ScriptError, NotImplementedError):
            raise
        except Exception as e:
            errors.append("{}: {}".format(line, e))
    if errors:
        raise ScriptError("Command bash {} returned an error: {}".format(path, "\n".join(errors)))
    return "".join(stdout)
