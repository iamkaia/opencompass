import argparse
import os
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from router_answer_supervision_core import (
    Collator,
    JointAnswerSupervisionRouterModel,
    build_lm_batch,
    build_dataset,
    discover_expert_names,
    save_json,
)


def parse_csv_arg(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    items = [part.strip() for part in str(raw).split(",") if part.strip()]
    return items or None


def build_lora_paths(args, expert_names: Sequence[str]) -> Dict[str, str]:
    all_lora_paths = {
        "iwslt2017": args.lora_iwslt,
        "medmcqa": args.lora_medmcqa,
        "race": args.lora_race,
        "squad2": args.lora_squad2,
        "sst2": args.lora_sst2,
        #"piqa": args.lora_piqa,
        #"copa": args.lora_copa,
        #"hellaswag": args.lora_hellaswag,
        #"boolq": args.lora_boolq,
        #"siqa": args.lora_siqa,
    }
    missing = [name for name in expert_names if not all_lora_paths.get(name)]
    if missing:
        raise ValueError(f"Missing LoRA paths for selected experts: {missing}")
    return {name: all_lora_paths[name] for name in expert_names}


def save_chunk(items: List[Dict], split_dir: str, chunk_idx: int, manifest_files: List[str]):
    filename = f"chunk_{chunk_idx:05d}.pt"
    torch.save({"items": items}, os.path.join(split_dir, filename))
    manifest_files.append(filename)


###建datasets的cache
def process_split(
    model: JointAnswerSupervisionRouterModel,
    split: str,
    data_root: str,
    requested_tasks: Optional[Sequence[str]],
    llm_tokenizer,
    output_root: str,
    batch_size: int,
    max_samples: Optional[int],
    max_llm_len: int,
    score_mode: str,
    add_eos_to_target: bool,
    seed: int,
    num_workers: int,
    chunk_size: int,
    feature_contract: Dict,
):
    dataset, task_names = build_dataset(
        data_root=data_root,
        split=split,
        requested_tasks=requested_tasks,
        expert_names=model.expert_names,
        max_samples=max_samples,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Collator(),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    split_dir = os.path.join(output_root, split)
    os.makedirs(split_dir, exist_ok=True)
    manifest_files: List[str] = []
    chunk_items: List[Dict] = []
    chunk_idx = 0

    device = next(model.parameters()).device
    num_tasks = len(model.expert_names)
    total_items = 0
    total_batches = len(loader)
    progress = tqdm(
        loader,
        total=total_batches,
        desc=f"cache:{split}",
        dynamic_ncols=True,
        mininterval=5.0,
        file=None,
    ) if tqdm is not None else loader
    print(
        f"[CACHE] start split={split} batches={total_batches} "
        f"tasks={task_names} num_experts={num_tasks}",
        flush=True,
    )

    for batch_idx, batch in enumerate(progress, start=1):
        '''
        prompt_input_ids
        prompt_attention_mask

        input_ids
        attention_mask
        labels

        prompt_input_ids:
        只有 prompt，用來抽 first_vec/mid_vec

        input_ids:
        prompt + target，用來算 target token NLL

        labels:
        prompt 部分是 -100，不算 loss
        target 部分才算 loss
        '''

        lm_batch = build_lm_batch(
            tokenizer=llm_tokenizer,
            prompts=batch.texts,
            targets=batch.targets,
            max_length=max_llm_len,
            add_eos_to_target=add_eos_to_target,
        )
        prompt_input_ids = lm_batch["prompt_input_ids"].to(device)
        prompt_attention_mask = lm_batch["prompt_attention_mask"].to(device)
        input_ids = lm_batch["input_ids"].to(device)
        attention_mask = lm_batch["attention_mask"].to(device)
        labels = lm_batch["labels"].to(device)

        with torch.no_grad():
            first_vec, mid_vec = model.extract_prompt_vectors(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
            )
            loss_matrix = model.score_all_route_pairs(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                targets=batch.targets,
                source_texts=batch.source_texts,
                task_names=batch.task_names,
                llm_tokenizer=llm_tokenizer,
                score_mode=score_mode,
            )
            correct_matrix = getattr(model, "last_route_correct_matrix", None)
            prediction_matrices = getattr(model, "last_route_prediction_matrices", None)

        flat_loss = loss_matrix.view(loss_matrix.size(0), -1)
        best_pair = flat_loss.argmin(dim=-1)
        best_first = best_pair // num_tasks
        best_mid = best_pair % num_tasks

        ####為甚麼都要給cpu?
        first_vec_cpu = first_vec.to(dtype=torch.float16).cpu()
        mid_vec_cpu = mid_vec.to(dtype=torch.float16).cpu()
        loss_matrix_cpu = loss_matrix.to(dtype=torch.float32).cpu()
        correct_matrix_cpu = (
            correct_matrix.to(dtype=torch.bool).cpu()
            if correct_matrix is not None
            else torch.zeros_like(loss_matrix, dtype=torch.bool).cpu()
        )
        best_pair_cpu = best_pair.cpu()
        best_first_cpu = best_first.cpu()
        best_mid_cpu = best_mid.cpu()
        task_ids_cpu = batch.task_ids.cpu()

        for idx in range(len(batch.texts)):
            prompt_text = batch.texts[idx]
            chunk_items.append(
                {
                    # Cache the same canonical prompt for both legacy `text`
                    # readers and newer `prompt_text` readers so cached BERT
                    # inputs and cached LLM prompt vectors are guaranteed to
                    # refer to the same string.
                    "text": prompt_text,
                    "source_text": prompt_text,
                    "prompt_text": prompt_text,
                    "original_source_text": batch.source_texts[idx],
                    "target": batch.targets[idx],
                    "task": batch.task_names[idx],
                    "task_id": int(task_ids_cpu[idx].item()),
                    "first_vec": first_vec_cpu[idx].clone(),
                    "mid_vec": mid_vec_cpu[idx].clone(),
                    "loss_matrix": loss_matrix_cpu[idx].clone(),
                    "correct_matrix": correct_matrix_cpu[idx].clone(),
                    "prediction_matrix": (
                        prediction_matrices[idx]
                        if prediction_matrices is not None
                        else None
                    ),
                    "pair_label": int(best_pair_cpu[idx].item()),
                    "first_label": int(best_first_cpu[idx].item()),
                    "mid_label": int(best_mid_cpu[idx].item()),
                }
            )
            total_items += 1

        if len(chunk_items) >= chunk_size:
            save_chunk(chunk_items, split_dir, chunk_idx, manifest_files)
            print(
                f"[CACHE] split={split} saved chunk={chunk_idx:05d} "
                f"items_in_chunk={len(chunk_items)} total_items={total_items}",
                flush=True,
            )
            chunk_items = []
            chunk_idx += 1

        if tqdm is not None:
            progress.set_postfix(
                batch=batch_idx,
                items=total_items,
                chunks=chunk_idx,
            )
            if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == total_batches:
                progress.write(
                    f"[CACHE] split={split} batch={batch_idx}/{total_batches} "
                    f"accumulated_items={total_items}"
                )
        elif batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == total_batches:
            print(
                f"[CACHE] split={split} batch={batch_idx}/{total_batches} "
                f"accumulated_items={total_items}",
                flush=True,
            )

    if chunk_items:
        save_chunk(chunk_items, split_dir, chunk_idx, manifest_files)
        print(
            f"[CACHE] split={split} saved final chunk={chunk_idx:05d} "
            f"items_in_chunk={len(chunk_items)} total_items={total_items}",
            flush=True,
        )
    if tqdm is not None:
        progress.close()

    save_json(
        {
            "split": split,
            "num_items": total_items,
            "files": manifest_files,
            "task_names": task_names,
            "expert_names": list(model.expert_names),
            "num_tasks": len(model.expert_names),
            "supervision_type": "cached_loss_matrix",
            "has_correct_matrix": True,
            "has_option_prob_matrix": False,
            "has_prediction_matrix": True,
            "score_mode": str(score_mode),
            "sst2_option_labels": "words",
            **feature_contract,
        },
        os.path.join(split_dir, "manifest.json"),
    )
    print(f"[CACHE] split={split} num_items={total_items} files={len(manifest_files)} out={split_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    ####最後輸出cache的地方
    parser.add_argument("--feature_root", type=str, required=True)
    ####資料來源要用哪些task來訓練router, 不填就用data_root底下所有的資料夾
    parser.add_argument("--task_names", type=str, default=None)
    ####建 cache 時枚舉哪些 expert
    parser.add_argument("--expert_names", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--router_bert_init", type=str, default ="./task_classifier_ckpt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=500)
    parser.add_argument("--max_val_samples", type=int, default=250)
    parser.add_argument("--max_llm_len", type=int, default=768)
    parser.add_argument("--first_layer_idx", type=int, default=0)
    parser.add_argument("--middle_layer_idx", type=int, default=15)
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument("--router_pooling", type=str, default="mean")
    parser.add_argument("--router_pooling_last_k", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="official_eval_aligned_generation",
        choices=["official_eval_aligned_generation", "official_generation_only"],
    )
    ####--add_eos_to_target又是甚麼意思? Ans. 就是label後面要不要加eos, 但現在好像沒有直接針對label的算score的方法，所以你可以等一下再看看
    parser.add_argument("--add_eos_to_target", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    ###chunk_size是甚麼意思?Ans. .pt cache 檔大約放多少筆 sample。

    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    ####有的時候不會所有expert都掛上去這樣可以嗎?
    parser.add_argument("--lora_iwslt", type=str, default='./saves/llama2-7b-chat-hf/lora/sft_iwslt')
    parser.add_argument("--lora_medmcqa", type=str, default='./saves/llama2-7b-chat-hf/lora/sft_medmcqa')
    parser.add_argument("--lora_race", type=str, default='./saves/llama2-7b-chat-hf/lora/sft_race')
    parser.add_argument("--lora_squad2", type=str, default='./saves/llama2-7b-chat-hf/lora/sft_squad20')
    parser.add_argument("--lora_sst2", type=str, default='./saves/llama2-7b-chat-hf/lora/sft_sst2')
    args = parser.parse_args()

    os.makedirs(args.feature_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    ###這次要拿哪些 tasks
    requested_tasks = parse_csv_arg(args.task_names)
    ###這次要拿哪些 expert
    requested_experts = parse_csv_arg(args.expert_names)
    all_lora_paths = {
        "iwslt2017": args.lora_iwslt,
        "medmcqa": args.lora_medmcqa,
        "race": args.lora_race,
        "squad2": args.lora_squad2,
        "sst2": args.lora_sst2,
    }
    expert_names, expert_name_source = discover_expert_names(requested_experts, all_lora_paths)
    lora_paths = build_lora_paths(args, expert_names)
    print(f"[INFO] expert_names={expert_names}")
    print(f"[INFO] expert_name_source={expert_name_source}")

    llm_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "left"
    '''
    1. 載 base LLM
    2. 把 LLM linear layer 換成 hard-routed LoRA linear
    3. 把五個 LoRA adapter 載進五個 expert slot
    4. 建 PromptVectorExtractor
    5. 建 BERT
    6. 建 router_first/router_mid/pair_classifier
    '''
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
    model.eval()
    feature_contract = {
        "feature_contract_version": 1,
        "first_layer_idx": int(args.first_layer_idx),
        "middle_layer_idx": int(args.middle_layer_idx),
        "router_pooling": str(args.router_pooling),
        "router_pooling_last_k": int(args.router_pooling_last_k),
        "llama_hidden_size": int(model.model.config.hidden_size),
    }

    save_json(
        {
            "data_root": args.data_root,
            "feature_root": args.feature_root,
            "task_names": requested_tasks or expert_names,
            "expert_names": expert_names,
            "base_model_path": args.base_model_path,
            "router_bert_init": args.router_bert_init,
            "max_llm_len": args.max_llm_len,
            "first_layer_idx": args.first_layer_idx,
            "middle_layer_idx": args.middle_layer_idx,
            "router_pooling": args.router_pooling,
            "router_pooling_last_k": args.router_pooling_last_k,
            "llama_hidden_size": int(model.model.config.hidden_size),
            "feature_contract_version": 1,
            "dtype": args.dtype,
            "score_mode": str(args.score_mode),
            "sst2_option_labels": "words",
            "chunk_size": args.chunk_size,
            "seed": args.seed,
        },
        os.path.join(args.feature_root, "cache_config.json"),
    )

    process_split(
        model=model,
        split="train",
        data_root=args.data_root,
        requested_tasks=requested_tasks,
        llm_tokenizer=llm_tokenizer,
        output_root=args.feature_root,
        batch_size=args.batch_size,
        max_samples=args.max_train_samples,
        max_llm_len=args.max_llm_len,
        score_mode=args.score_mode,
        add_eos_to_target=args.add_eos_to_target,
        seed=args.seed,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        feature_contract=feature_contract,
    )
    process_split(
        model=model,
        split="validation",
        data_root=args.data_root,
        requested_tasks=requested_tasks,
        llm_tokenizer=llm_tokenizer,
        output_root=args.feature_root,
        batch_size=args.batch_size,
        max_samples=args.max_val_samples,
        max_llm_len=args.max_llm_len,
        score_mode=args.score_mode,
        add_eos_to_target=args.add_eos_to_target,
        seed=args.seed,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        feature_contract=feature_contract,
    )
    print("[DONE] cached dataset build finished")


if __name__ == "__main__":
    main()
