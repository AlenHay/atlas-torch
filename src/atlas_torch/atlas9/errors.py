"""Program termination as the Fortran does it (STOP / CALL EXIT / runtime end-of-file)."""


class FortranExit(Exception):
    """The Fortran program terminates. `message` is what it printed last (list-directed, e.g. 'HYDROFAIL',
    'REACHED GOLD TOLERANCES', 'MODEL DIVERGED', 'INVALID STRUCTURE', or '' for a bare CALL EXIT), `code`
    the process exit status (0 for STOP / CALL EXIT, 2 for a gfortran runtime error such as end of file)."""

    def __init__(self, message: str = "", code: int = 0):
        super().__init__(message)
        self.message = message
        self.code = code
