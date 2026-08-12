import json
from app.domain.models.reasoning_context import ReasoningContext
from app.domain.models.reasoning import EngineResult
from app.agents.reasoning.prompts import EXPLANATION_GENERATOR_SYSTEM_PROMPT, EXPLANATION_GENERATOR_USER_PROMPT

class ExplanationGenerator:
    """
    Semantic LLM component.
    Translates ranked hypotheses into narrative.
    """
    def __init__(self, llm_client=None):
        self.llm = llm_client

    async def run(self, context: ReasoningContext, ranked_hypotheses: list[dict]) -> EngineResult:
        explanation = ""
        
        if self.llm and ranked_hypotheses:
            try:
                top = ranked_hypotheses[0]
                alts = ranked_hypotheses[1:]
                
                trace_str = json.dumps(context.trace.logs, default=str)
                
                response = await self.llm.ainvoke([
                    {"role": "system", "content": EXPLANATION_GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": EXPLANATION_GENERATOR_USER_PROMPT.format(
                        top_hypothesis=json.dumps(top, default=str),
                        alternative_hypotheses=json.dumps(alts, default=str),
                        trace=trace_str
                    )}
                ])
                raw_content = response.content.strip()
                print(f"\\n--- LLM EXPLANATION RAW OUTPUT ---\\n{raw_content}\\n-----------------------------------\\n")
                explanation = raw_content
            except Exception as e:
                return EngineResult(success=False, errors=[str(e)])
        else:
            if ranked_hypotheses:
                explanation = f"Top hypothesis: {ranked_hypotheses[0].get('statement', 'Unknown')}"
            else:
                explanation = "The available CCTV evidence is insufficient to answer your query."

        return EngineResult(
            success=True,
            partial_output={"explanation": explanation}
        )
