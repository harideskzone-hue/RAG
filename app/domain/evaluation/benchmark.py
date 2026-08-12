from app.domain.evaluation.manifest import EvaluationManifest
from app.domain.evaluation.suite import BenchmarkSuite
from app.domain.evaluation.evaluator import Evaluator
from app.domain.evaluation.baseline import BaselineRepository
from app.domain.evaluation.score import EvaluationScore
from app.domain.evaluation.regression import RegressionDetector
from app.domain.evaluation.result import EvaluationResult
from app.domain.evaluation.statistics import BenchmarkStatistics
import time
from uuid import uuid4

class BenchmarkRunner:
    """The CLI runner that loads suites and runs the Evaluator based on Manifest."""
    def __init__(self, manifest: EvaluationManifest, evaluator: Evaluator):
        self.manifest = manifest
        self.evaluator = evaluator
        
    def run(self, suite: BenchmarkSuite) -> EvaluationResult:
        start_time = time.time()
        
        stats = BenchmarkStatistics()
        traces = []
        
        for dataset in suite.datasets:
            if dataset.version != self.manifest.dataset_version:
                continue
                
            for test_case in dataset.test_cases:
                trace, metrics = self.evaluator.evaluate(test_case)
                traces.append(trace)
                stats.total_tests_run += 1
                stats.passed_tests += 1
                stats.aggregate_metrics = metrics # simplified aggregation for MVP
                
        # 3. Score
        final_score = EvaluationScore.compute(stats.aggregate_metrics)
        
        # 4. Check Regressions
        baseline = BaselineRepository.get_baseline(self.manifest.baseline_version)
        regression = RegressionDetector.check_regression(final_score, baseline, self.manifest.fail_threshold)
        
        duration = (time.time() - start_time) * 1000
        
        result = EvaluationResult(
            run_id=str(uuid4()),
            metrics=stats.aggregate_metrics,
            scores=final_score,
            regression=regression,
            traces=traces,
            duration_ms=duration
        )
        
        return result
