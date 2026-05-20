"""Evaluation, benchmark, and manual CV orchestration."""

from .benchmark_analysis import (
    build_analysis,
    load_benchmark_rows,
    write_analysis_outputs,
)
from .benchmark_matrix import (
    BENCHMARK_CSV_FIELDNAMES,
    iter_benchmark_jobs,
    job_id,
    load_benchmark_datasets,
    run_benchmark_job,
    run_benchmark_jobs,
    select_specs,
)
from .fit_methods import (
    FitMethodSpec,
    create_fit_method_specs,
    fit_method_specs_by_key,
    run_fit_method,
    run_grid_search_fit_method,
    run_manual_loo_fit_method,
    run_train_test_fit_method,
)
from .results_io import (
    initialize_output_csv,
    read_completed_ids,
    write_rows_csv,
)
