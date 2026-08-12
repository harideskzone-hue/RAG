from app.domain.evaluation.result import EvaluationResult

class ReportGenerator:
    """Generates comprehensive multi-format reports."""
    
    @staticmethod
    def generate_console(result: EvaluationResult) -> str:
        out = f"=== Evaluation Report [{result.run_id}] ===\n"
        out += f"Score: {result.scores.overall_score:.1f}\n"
        out += f"Regression: {result.regression.message}\n"
        out += f"Duration: {result.duration_ms:.1f}ms\n"
        return out
        
    @staticmethod
    def generate_markdown(result: EvaluationResult) -> str:
        return f"# Evaluation Report\n\n**Score**: {result.scores.overall_score:.1f}\n\n**Regression**: {result.regression.message}\n"
        
    @staticmethod
    def generate_json(result: EvaluationResult) -> str:
        return result.model_dump_json(indent=2)
        
    @staticmethod
    def generate_junit(result: EvaluationResult) -> str:
        failure = f"<failure message='{result.regression.message}'/>" if result.regression.is_regression else ""
        return f"<testsuites>\n  <testsuite name='VISTA_Eval' time='{result.duration_ms/1000}'>\n    <testcase name='Overall Score'>\n      {failure}\n    </testcase>\n  </testsuite>\n</testsuites>"
