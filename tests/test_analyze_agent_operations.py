import json
from pathlib import Path

from edgemed_bench.analyze_agent_operations import analyze_agent_operations
from edgemed_bench.medical_agent_fixture import run_fixture


def test_operational_analysis_is_reference_free_and_gates_targeted_roi(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_fixture(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"status": "inference_completed", "completed_total": 1})
    manifest_path.write_text(json.dumps(manifest) + "\n")
    references = run_dir / "references.jsonl"
    references.unlink()
    report = analyze_agent_operations(
        run_dir, target_area_min=0.01, target_area_max=0.64, target_rate_min=0.5
    )
    assert report["overall"] == "PASS"
    assert report["e0_structure"] == {
        "schema_valid_rate": 1.0,
        "citation_valid_rate": 1.0,
        "tool_trace_bound_rate": 1.0,
    }
    assert report["tools"]["targeted_sample_rate"] == 1.0
    assert report["tools"]["failed"] == 0
    assert "references_sha256" not in report["source_hashes"]


def test_operational_analysis_blocks_full_frame_region(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_fixture(run_dir)
    traces_path = run_dir / "tool_traces.jsonl"
    traces = [json.loads(line) for line in traces_path.read_text().splitlines()]
    for trace in traces:
        if trace["tool_name"] == "region_inspect":
            trace["request"]["region_xyxy_1000"] = [0, 0, 1000, 1000]
    traces_path.write_text("".join(json.dumps(row) + "\n" for row in traces))
    report = analyze_agent_operations(run_dir, target_rate_min=0.5)
    assert report["overall"] == "BLOCK"
    assert report["gates"]["targeted_roi_rate"] is False
