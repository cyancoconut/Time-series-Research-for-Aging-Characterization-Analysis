"""What varies between a check-up run and a characterization run.

The pipeline in :mod:`main` is shared. Three things differ:

* which BRONZE layer supplies the payload (``BRONZE_CU`` vs ``BRONZE_PARA``),
* which config key names the procedure filter that selects the test files,
* where every artefact is written.

They travel together as one explicit parameter rather than as config keys, so
any function's output destination is readable from its own signature.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunContext:
    #: BRONZE layer folder / object prefix the run reads.
    bronze_layer: str = "BRONZE_CU"
    #: Battery-config key holding the procedure filter for this run.
    procedure_filter_key: str = "procedure_filter"
    #: When set, every artefact goes under `<prefix>/<cell_stem>/` instead of
    #: the shared GOLD/20_/25_/30_/40_ folders. MinIO keys are untagged.
    export_root_prefix: str | None = None
    #: Run the pulse/EIS/qOCV exports regardless of the config's export_* flags.
    force_exports: bool = False

    def export_root(self, cell: str) -> str | None:
        """Per-cell output root as a POSIX-style relative path, or None."""
        if not self.export_root_prefix:
            return None
        return f"{self.export_root_prefix}/{cell.split('.')[0]}"


#: The check-up pipeline — today's behaviour in every respect.
CU = RunContext()

#: Initial characterization (BOL parametrization).
CHARACTERIZATION = RunContext(
    bronze_layer="BRONZE_PARA",
    procedure_filter_key="para_procedure_filter",
    export_root_prefix="10_initial_characterization",
    force_exports=True,
)
