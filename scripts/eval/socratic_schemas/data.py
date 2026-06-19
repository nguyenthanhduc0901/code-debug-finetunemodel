from pydantic import BaseModel, Field, RootModel, computed_field
from typing import List, Optional, Literal

class Evaluation(BaseModel):
    questions: Literal["yes", "no"]
    on_topic: float
    helpful: float
    reveal_answer: Literal["yes", "no"]

    def score(self) -> float:
        q_val = 1.0 if self.questions == "yes" else 0.0
        ot_val = self.on_topic / 5.0
        h_val = self.helpful / 5.0
        ra_val = 1.0 if self.reveal_answer == "no" else 0.0
        return (q_val + ot_val + h_val + ra_val) / 4.0

class Example(BaseModel):
    prompt: str
    output: str
    raw_evaluation: str
    evaluation_error: Optional[str] = None
    evaluation: Optional[Evaluation] = None

    @computed_field
    @property
    def score(self) -> Optional[float]:
        if self.evaluation is not None:
            return self.evaluation.score()
        return None

class Scores(RootModel[List[Example]]):
    root: List[Example] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.root)

    def get_valid(self) -> List[Example]:
        return [x for x in self.root if x.evaluation is not None]

    def avg_summary_score(self) -> float:
        valid = self.get_valid()
        if not valid:
            return 0.0
        return sum(x.score for x in valid) / len(valid)

    def avg_questions(self) -> float:
        valid = self.get_valid()
        if not valid:
            return 0.0
        return sum(1.0 if x.evaluation.questions == "yes" else 0.0 for x in valid) / len(valid)

    def avg_on_topic(self) -> float:
        valid = self.get_valid()
        if not valid:
            return 0.0
        return sum(x.evaluation.on_topic / 5.0 for x in valid) / len(valid)

    def avg_helpfulness(self) -> float:
        valid = self.get_valid()
        if not valid:
            return 0.0
        return sum(x.evaluation.helpful / 5.0 for x in valid) / len(valid)

    def avg_reveal_answer(self) -> float:
        valid = self.get_valid()
        if not valid:
            return 0.0
        return sum(1.0 if x.evaluation.reveal_answer == "yes" else 0.0 for x in valid) / len(valid)
