import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from opencompass.registry import TEXT_POSTPROCESSORS
from opencompass.utils.text_postprocessors import (
    first_capital_postprocess,
    first_option_postprocess,
    general_cn_postprocess,
    sst2_postprocess,
)


@dataclass(frozen=True)
class TaskEvalSpec:
    score_family: str
    evaluator_key: str
    pred_postprocessor: Optional[Dict[str, Any]] = None
    dataset_postprocessor: Optional[Dict[str, Any]] = None
    reference_adapter: Optional[Callable[[str, str], Any]] = None
    use_raw_prediction: bool = False
    extra_score_kwargs_builder: Optional[Callable[[str, str], Dict[str, Any]]] = None


def normalize_qa_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", text)
    return text


def normalize_sst2_label(text: str) -> str:
    value = str(text).strip().lower()
    if value in {"1", "positive"}:
        return "1"
    if value in {"0", "negative"}:
        return "0"
    return value


def normalize_boolq_label(text: str) -> str:
    value = str(text).strip().lower()
    if value in {"a", "yes", "true", "1"}:
        return "A"
    if value in {"b", "no", "false", "0"}:
        return "B"
    return value.upper()[:1]


def parse_squad_references(target: str) -> List[str]:
    raw = target
    if isinstance(raw, (list, tuple)):
        refs = [str(item) for item in raw]
    else:
        refs = []
        text = str(raw).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    refs = [str(item) for item in parsed]
            except Exception:
                refs = []
        if not refs:
            refs = [text]
    refs = [ref for ref in refs if str(ref).strip()]
    return refs or [str(target).strip()]


def split_option_block(text: str, labels: Sequence[str]) -> Dict[str, str]:
    positions = []
    for label in labels:
        match = re.search(rf"{re.escape(label)}[\.:]\s*", str(text))
        if match:
            positions.append((label, match.start(), match.end()))
    if len(positions) != len(labels):
        return {}
    positions.sort(key=lambda item: item[1])
    parsed = {}
    for idx, (label, _start, content_start) in enumerate(positions):
        next_start = positions[idx + 1][1] if idx + 1 < len(positions) else len(text)
        parsed[label] = str(text)[content_start:next_start].strip()
    return parsed


def parse_siqa_reference(source_text: str, target: str) -> Optional[Dict[str, Any]]:
    parsed = split_option_block(source_text, ["A", "B", "C"])
    if not parsed:
        return None
    label = str(target).strip().upper()[:1]
    if label not in parsed:
        return None
    return {
        "candidates": [
            [f"A. {parsed['A']}", "A", parsed["A"]],
            [f"B. {parsed['B']}", "B", parsed["B"]],
            [f"C. {parsed['C']}", "C", parsed["C"]],
        ],
        "label": ord(label) - ord("A"),
    }


def build_medmcqa_score_kwargs(source_text: str, target: str) -> Dict[str, Any]:
    options = split_option_block(source_text, ["A", "B", "C", "D"])
    gold = str(target).strip().upper()[:1]
    gold_idx = max(0, ord(gold) - ord("A")) if gold else 0
    if not options:
        return {
            "references": [gold_idx],
            "test_set": {
                "prompt_mode": ["zero-shot"],
                "options": [["", "", "", ""]],
                "label": [gold],
                "subject_name": [""],
                "topic_name": [""],
                "choice_type": [""],
            },
        }
    return {
        "references": [gold_idx],
        "test_set": {
            "prompt_mode": ["zero-shot"],
            "options": [[options["A"], options["B"], options["C"], options["D"]]],
            "label": [gold],
            "subject_name": [""],
            "topic_name": [""],
            "choice_type": [""],
        },
    }


def build_race_score_kwargs(source_text: str, _target: str) -> Dict[str, Any]:
    return {"origin_prompt": [str(source_text or "")]}


def resolve_postprocessor(proc_spec: Dict[str, Any]):
    kwargs = dict(proc_spec)
    proc = kwargs.pop("type")
    if isinstance(proc, str):
        proc = TEXT_POSTPROCESSORS.get(proc)
    return proc, kwargs


def apply_postprocessor(values: Sequence[str], proc_spec: Optional[Dict[str, Any]]) -> List[str]:
    if not proc_spec:
        return [str(value) for value in values]
    proc, kwargs = resolve_postprocessor(proc_spec)
    return [proc(str(value), **kwargs) for value in values]


TASK_EVAL_SPECS: Dict[str, TaskEvalSpec] = {
    "race": TaskEvalSpec(
        score_family="accuracy",
        evaluator_key="acc_with_details",
        pred_postprocessor={"type": first_option_postprocess, "options": "ABCD"},
        extra_score_kwargs_builder=build_race_score_kwargs,
    ),
    "medmcqa": TaskEvalSpec(
        score_family="accuracy",
        evaluator_key="medmcqa",
        extra_score_kwargs_builder=build_medmcqa_score_kwargs,
        use_raw_prediction=True,
    ),
    "hellaswag": TaskEvalSpec(
        score_family="accuracy",
        evaluator_key="acc",
        pred_postprocessor={"type": first_option_postprocess, "options": "ABCD"},
    ),
    "piqa": TaskEvalSpec(
        score_family="accuracy",
        evaluator_key="acc",
        pred_postprocessor={"type": first_option_postprocess, "options": "AB"},
    ),
    "copa": TaskEvalSpec(
        score_family="accuracy",
        evaluator_key="acc",
        pred_postprocessor={"type": first_option_postprocess, "options": "AB"},
    ),
    "sst2": TaskEvalSpec(
        score_family="accuracy",
        evaluator_key="acc",
        pred_postprocessor={"type": sst2_postprocess},
    ),
    "boolq": TaskEvalSpec(
        score_family="accuracy",
        evaluator_key="acc",
        pred_postprocessor={"type": first_capital_postprocess},
    ),
    "siqa": TaskEvalSpec(
        score_family="accuracy",
        evaluator_key="edacc",
        reference_adapter=parse_siqa_reference,
        use_raw_prediction=True,
    ),
    "squad2": TaskEvalSpec(
        score_family="score",
        evaluator_key="squad20",
        reference_adapter=lambda _source, target: parse_squad_references(target),
        use_raw_prediction=True,
    ),
    "squad20": TaskEvalSpec(
        score_family="score",
        evaluator_key="squad20",
        reference_adapter=lambda _source, target: parse_squad_references(target),
        use_raw_prediction=True,
    ),
    "squad2.0": TaskEvalSpec(
        score_family="score",
        evaluator_key="squad20",
        reference_adapter=lambda _source, target: parse_squad_references(target),
        use_raw_prediction=True,
    ),
    "iwslt2017": TaskEvalSpec(
        score_family="bleu",
        evaluator_key="bleu",
        pred_postprocessor={"type": general_cn_postprocess},
        dataset_postprocessor={"type": general_cn_postprocess},
    ),
}
