import argparse
import json
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoTokenizer

NULL_EXPERT_ID = None
set_all_experts = None
set_layer_range_expert = None
build_lm_batch = None


def truncate_text(text: str, limit: int) -> str:
    text = str(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated {len(text) - limit} chars]"


def pair_rows(expert_names: Sequence[str]) -> List[Dict]:
    rows = []
    num_tasks = len(expert_names)
    for pair_idx in range(num_tasks * num_tasks):
        first_idx = pair_idx // num_tasks
        mid_idx = pair_idx % num_tasks
        rows.append(
            {
                "pair_idx": pair_idx,
                "first_idx": first_idx,
                "mid_idx": mid_idx,
                "first_expert": expert_names[first_idx],
                "mid_expert": expert_names[mid_idx],
                "name": f"{expert_names[first_idx]}->{expert_names[mid_idx]}",
            }
        )
    return rows


@torch.no_grad()
def generate_for_pair(
    model,
    tokenizer,
    prompt: str,
    first_task: str,
    mid_task: str,
    max_llm_len: int,
    max_new_tokens: int,
) -> str:
    lm_batch = build_lm_batch(
        tokenizer=tokenizer,
        prompts=[prompt],
        targets=[""],
        max_length=max_llm_len,
        add_eos_to_target=False,
    )
    prompt_input_ids = lm_batch["prompt_input_ids"].to(next(model.parameters()).device)
    prompt_attention_mask = lm_batch["prompt_attention_mask"].to(next(model.parameters()).device)

    first_eid = model.task_to_expert_id[first_task]
    mid_eid = model.task_to_expert_id[mid_task]
    set_all_experts(model.model, NULL_EXPERT_ID)
    set_layer_range_expert(model.model, model.first_layer_idx, model.middle_layer_idx - 1, first_eid)
    set_layer_range_expert(model.model, model.middle_layer_idx, model.num_layers - 1, mid_eid)

    outputs = model.model.generate(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    set_all_experts(model.model, NULL_EXPERT_ID)
    new_tokens = outputs[0, prompt_input_ids.size(1) :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.no_grad()
def predict_router_pair(
    model,
    llm_tokenizer,
    bert_tokenizer,
    llama_prompt: str,
    bert_prompt: str,
    max_llm_len: int,
    max_bert_len: int,
) -> Optional[Dict]:
    if not hasattr(model, "forward_router"):
        return None

    lm_batch = build_lm_batch(
        tokenizer=llm_tokenizer,
        prompts=[llama_prompt],
        targets=[""],
        max_length=max_llm_len,
        add_eos_to_target=False,
    )
    prompt_input_ids = lm_batch["prompt_input_ids"].to(next(model.parameters()).device)
    prompt_attention_mask = lm_batch["prompt_attention_mask"].to(next(model.parameters()).device)
    first_vec, mid_vec = model.extract_prompt_vectors(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
    )

    bert_enc = bert_tokenizer(
        [bert_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_bert_len,
    )
    bert_input_ids = bert_enc["input_ids"].to(next(model.parameters()).device)
    bert_attention_mask = bert_enc["attention_mask"].to(next(model.parameters()).device)
    bert_token_type_ids = bert_enc.get("token_type_ids")
    if bert_token_type_ids is not None:
        bert_token_type_ids = bert_token_type_ids.to(next(model.parameters()).device)

    pair_logits, _, _ = model.forward_router(
        bert_input_ids=bert_input_ids,
        bert_attention_mask=bert_attention_mask,
        bert_token_type_ids=bert_token_type_ids,
        first_vec=first_vec,
        mid_vec=mid_vec,
    )
    probs = torch.softmax(pair_logits.float(), dim=-1)[0]
    pred_pair = int(probs.argmax().item())
    num_tasks = len(model.expert_names)
    return {
        "pred_pair": pred_pair,
        "pred_first": pred_pair // num_tasks,
        "pred_mid": pred_pair % num_tasks,
        "pred_prob": float(probs[pred_pair].item()),
    }


def maybe_load_router_weights(model, ckpt_dir: Optional[str]):
    if not ckpt_dir:
        return
    model.load_router_weights(ckpt_dir)


def parse_csv_arg(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    items = [part.strip() for part in str(raw).split(",") if part.strip()]
    return items or None


def discover_expert_names_from_lora(
    requested_experts: Optional[Sequence[str]],
    all_lora_paths: Dict[str, Optional[str]],
) -> List[str]:
    if requested_experts:
        return [str(name).strip() for name in requested_experts if str(name).strip()]
    expert_names = [name for name, path in all_lora_paths.items() if path]
    if not expert_names:
        return ["iwslt2017", "medmcqa", "race", "squad2", "sst2"]
    return expert_names


def build_lora_paths_from_args(args, expert_names: Sequence[str]) -> Dict[str, str]:
    all_lora_paths = {
        "iwslt2017": args.lora_iwslt,
        "medmcqa": args.lora_medmcqa,
        "race": args.lora_race,
        "squad2": args.lora_squad2,
        "sst2": args.lora_sst2,
        "piqa": args.lora_piqa,
        "copa": args.lora_copa,
        "hellaswag": args.lora_hellaswag,
        "boolq": args.lora_boolq,
        "siqa": args.lora_siqa,
    }
    missing = [name for name in expert_names if not all_lora_paths.get(name)]
    if missing:
        raise ValueError(f"Missing LoRA paths for selected experts: {missing}")
    return {name: all_lora_paths[name] for name in expert_names}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print the exact LLaMA prompt, BERT prompt, and generation from every "
            "first/mid expert pair for sampled router data."
        )
    )
    parser.add_argument("--data_root", type=str, default="router_datasets_5org_train25")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--expert_names", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--router_bert_init", type=str, default="./task_classifier_ckpt")
    parser.add_argument("--router_ckpt_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_llm_len", type=int, default=768)
    parser.add_argument("--max_bert_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--first_layer_idx", type=int, default=0)
    parser.add_argument("--middle_layer_idx", type=int, default=15)
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument("--router_pooling", type=str, default="mean")
    parser.add_argument("--router_pooling_last_k", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--text_limit", type=int, default=2000)
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument(
        "--output_pretty_json",
        type=str,
        default=None,
        help="Write a human-readable JSON array where each sample is pretty-printed on separate lines.",
    )

    parser.add_argument("--lora_iwslt", type=str, default="./saves/llama2-7b-chat-hf/lora/sft_iwslt")
    parser.add_argument("--lora_medmcqa", type=str, default="./saves/llama2-7b-chat-hf/lora/sft_medmcqa")
    parser.add_argument("--lora_race", type=str, default="./saves/llama2-7b-chat-hf/lora/sft_race")
    parser.add_argument("--lora_squad2", type=str, default="./saves/llama2-7b-chat-hf/lora/sft_squad20")
    parser.add_argument("--lora_sst2", type=str, default="./saves/llama2-7b-chat-hf/lora/sft_sst2")
    parser.add_argument("--lora_piqa", type=str, default=None)
    parser.add_argument("--lora_copa", type=str, default=None)
    parser.add_argument("--lora_hellaswag", type=str, default=None)
    parser.add_argument("--lora_boolq", type=str, default=None)
    parser.add_argument("--lora_siqa", type=str, default=None)
    args = parser.parse_args()

    global NULL_EXPERT_ID, set_all_experts, set_layer_range_expert, build_lm_batch
    from train_joint_answer_supervision_router import (
        JointAnswerSupervisionRouterModel,
        build_dataset,
        build_lm_batch as imported_build_lm_batch,
    )
    from opencompass.models.router_moe_shared import (
        NULL_EXPERT_ID as imported_null_expert_id,
        set_all_experts as imported_set_all_experts,
        set_layer_range_expert as imported_set_layer_range_expert,
    )

    NULL_EXPERT_ID = imported_null_expert_id
    set_all_experts = imported_set_all_experts
    set_layer_range_expert = imported_set_layer_range_expert
    build_lm_batch = imported_build_lm_batch

    requested_tasks = parse_csv_arg(args.task_names)
    requested_experts = parse_csv_arg(args.expert_names)
    all_lora_paths = {
        "iwslt2017": args.lora_iwslt,
        "medmcqa": args.lora_medmcqa,
        "race": args.lora_race,
        "squad2": args.lora_squad2,
        "sst2": args.lora_sst2,
        "piqa": args.lora_piqa,
        "copa": args.lora_copa,
        "hellaswag": args.lora_hellaswag,
        "boolq": args.lora_boolq,
        "siqa": args.lora_siqa,
    }
    expert_names = discover_expert_names_from_lora(requested_experts, all_lora_paths)
    lora_paths = build_lora_paths_from_args(args, expert_names)

    dataset, task_names = build_dataset(
        data_root=args.data_root,
        split=args.split,
        requested_tasks=requested_tasks,
        max_samples=args.num_samples,
        seed=args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    llm_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "left"
    bert_tokenizer = AutoTokenizer.from_pretrained(args.router_bert_init)

    model = JointAnswerSupervisionRouterModel(
        base_model_path=args.base_model_path,
        router_bert_init=args.router_bert_init,
        lora_paths=lora_paths,
        first_layer_idx=args.first_layer_idx,
        middle_layer_idx=args.middle_layer_idx,
        router_dim=args.router_dim,
        dtype=args.dtype,
        r=args.r,
        alpha=args.alpha,
        expert_names=expert_names,
        router_pooling=args.router_pooling,
        router_pooling_last_k=args.router_pooling_last_k,
    ).to(device)
    maybe_load_router_weights(model, args.router_ckpt_dir)
    model.eval()

    pair_table = pair_rows(expert_names)
    print(f"[INFO] task_names={task_names}")
    print(f"[INFO] expert_names={expert_names}")
    print(f"[INFO] num_pairs={len(pair_table)}")
    for row in pair_table:
        print(
            f"[PAIR] {row['pair_idx']:02d}: first={row['first_expert']} "
            f"mid={row['mid_expert']} name={row['name']}"
        )

    out_f = open(args.output_jsonl, "w", encoding="utf-8") if args.output_jsonl else None
    pretty_records = []
    try:
        for sample_idx in range(min(args.num_samples, len(dataset))):
            item = dataset[sample_idx]
            llama_prompt = item["text"]
            bert_prompt = item["source_text"]
            record = {
                "sample_idx": sample_idx,
                "task": item["task"],
                "target": item["target"],
                "llama_prompt": llama_prompt,
                "bert_prompt": bert_prompt,
                "pairs": [],
            }

            print("\n" + "=" * 100)
            print(f"[SAMPLE] idx={sample_idx} task={item['task']} target={item['target']!r}")
            print("[LLAMA_PROMPT]")
            print(truncate_text(llama_prompt, args.text_limit))
            print("[BERT_PROMPT]")
            print(truncate_text(bert_prompt, args.text_limit))

            router_pred = predict_router_pair(
                model=model,
                llm_tokenizer=llm_tokenizer,
                bert_tokenizer=bert_tokenizer,
                llama_prompt=llama_prompt,
                bert_prompt=bert_prompt,
                max_llm_len=args.max_llm_len,
                max_bert_len=args.max_bert_len,
            )
            if router_pred is not None and args.router_ckpt_dir:
                first_name = expert_names[router_pred["pred_first"]]
                mid_name = expert_names[router_pred["pred_mid"]]
                record["router_prediction"] = {
                    **router_pred,
                    "name": f"{first_name}->{mid_name}",
                }
                print(
                    f"[ROUTER_PRED] pair={router_pred['pred_pair']:02d} "
                    f"name={first_name}->{mid_name} prob={router_pred['pred_prob']:.6f}"
                )

            for pair in pair_table:
                generated = generate_for_pair(
                    model=model,
                    tokenizer=llm_tokenizer,
                    prompt=llama_prompt,
                    first_task=pair["first_expert"],
                    mid_task=pair["mid_expert"],
                    max_llm_len=args.max_llm_len,
                    max_new_tokens=args.max_new_tokens,
                )
                pair_record = dict(pair)
                pair_record["output"] = generated
                record["pairs"].append(pair_record)
                print(f"[OUTPUT][{pair['pair_idx']:02d}][{pair['name']}] {generated!r}")

            if out_f is not None:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
            if args.output_pretty_json:
                pretty_records.append(record)
    finally:
        if out_f is not None:
            out_f.close()
        if args.output_pretty_json:
            with open(args.output_pretty_json, "w", encoding="utf-8") as f:
                json.dump(pretty_records, f, ensure_ascii=False, indent=2)
                f.write("\n")


if __name__ == "__main__":
    main()
