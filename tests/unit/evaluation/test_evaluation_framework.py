import pytest
from app.domain.evaluation.testcase import TestCase
from app.domain.evaluation.dataset import EvaluationDataset
from app.domain.evaluation.suite import BenchmarkSuite
from app.domain.evaluation.manifest import EvaluationManifest
from app.domain.evaluation.profile import EvaluationProfile
from app.domain.evaluation.planner_scorer import PlannerScorer
from app.domain.evaluation.graph_scorer import GraphScorer
from app.domain.evaluation.evaluator import Evaluator
from app.domain.evaluation.benchmark import BenchmarkRunner
from app.domain.evaluation.report import ReportGenerator

def test_evaluation_framework():
    # 1. Setup Data
    test_case = TestCase(
        test_id="tc_1",
        description="Find person",
        query="Who is this?",
        expected_agents=["metadata_agent", "video_agent"]
    )
    dataset = EvaluationDataset(
        dataset_id="ds_1",
        version="v1",
        description="Smoke Dataset",
        test_cases=[test_case]
    )
    suite = BenchmarkSuite(
        suite_id="suite_1",
        name="Smoke Suite",
        datasets=[dataset]
    )
    
    # 2. Setup Config
    manifest = EvaluationManifest(
        enabled_scorers=["planner_scorer", "graph_scorer"],
        required_profiles=[EvaluationProfile.FAST],
        dataset_version="v1",
        baseline_version="v1"
    )
    
    # 3. Setup Evaluator
    scorers = [PlannerScorer(), GraphScorer()]
    evaluator = Evaluator(scorers)
    
    # 4. Run Benchmark
    runner = BenchmarkRunner(manifest, evaluator)
    result = runner.run(suite)
    
    # 5. Assertions
    assert result.scores.overall_score > 0
    assert result.regression.is_regression is False
    assert len(result.traces) == 1
    
    # 6. Report
    console_report = ReportGenerator.generate_console(result)
    assert "Evaluation Report" in console_report
    assert "Score:" in console_report
