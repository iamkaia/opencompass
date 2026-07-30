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
from model_backbone_specs import get_hidden_size


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


def preview_text(text: str, limit: int = 320) -> str:
    text = str(text).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def model_uses_qwen_chat_template(model_path: str) -> bool:
    return "qwen" in str(model_path).lower()


def normalize_chat_template_content(prompt: str, model_path: str) -> str:
    text = str(prompt)
    # OpenCompass PromptTemplate leaves a newline after the user round before
    # Qwen's <|im_end|>. Cache the same text so train-time and eval-time
    # routing see the same message boundary.
    if model_uses_qwen_chat_template(model_path) and not text.endswith("\n"):
        return text + "\n"
    return text


def apply_cache_prompt_template(prompt: str, tokenizer, mode: str, model_path: str) -> str:
    if mode == "raw":
        return str(prompt)
    if mode != "chat_template":
        raise ValueError(f"Unsupported cache_prompt_template={mode!r}")
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "--cache_prompt_template chat_template requires a tokenizer with apply_chat_template"
        )
    content = normalize_chat_template_content(prompt, model_path)
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


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
    cache_prompt_template: str,
    base_model_path: str,
    print_cache_prompt_examples: bool,
    route_space: str,
):
    ###1. 讀資料
    dataset, task_names = build_dataset(
        data_root=data_root,
        split=split,
        requested_tasks=requested_tasks,
        expert_names=model.expert_names,
        max_samples=max_samples,
        seed=seed,
    )
    
    ###2. 建 DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Collator(),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    ###3. 建輸出資料夾
    split_dir = os.path.join(output_root, split)
    os.makedirs(split_dir, exist_ok=True)
    manifest_files: List[str] = []
    chunk_items: List[Dict] = []
    chunk_idx = 0

    device = next(model.parameters()).device
    num_tasks = len(model.expert_names)
    total_items = 0
    printed_prompt_tasks = set()
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

        
        ###4. 對每個 batch 做 prompt template
        '''
        cache_prompt_texts = [
            apply_cache_prompt_template(...)
        ]

        如果你用：

        --cache_prompt_template raw

        就原樣使用 prompt。

        如果用：

        --cache_prompt_template chat_template

        就會套 tokenizer 的 chat template，讓 cache prompt 更接近 OpenCompass runtime。
        '''
        cache_prompt_texts = [
            apply_cache_prompt_template(text, llm_tokenizer, cache_prompt_template, base_model_path)
            for text in batch.texts
        ]
        if print_cache_prompt_examples:
            for task_name, raw_text, cache_text in zip(batch.task_names, batch.texts, cache_prompt_texts):
                if task_name in printed_prompt_tasks:
                    continue
                printed_prompt_tasks.add(task_name)
                print(
                    f"[CACHE_PROMPT][split={split}][task={task_name}] "
                    f"template={cache_prompt_template} raw_len={len(str(raw_text))} "
                    f"cache_len={len(str(cache_text))} "
                    f"raw={preview_text(raw_text)} cache={preview_text(cache_text)}",
                    flush=True,
                )



        ###5. tokenize prompt + target
        '''
        產生：
        prompt_input_ids
        prompt_attention_mask
        input_ids
        attention_mask
        labels

        其中：

        prompt_input_ids:
            只含 prompt，用來抽 first_vec / mid_vec

        input_ids:
            prompt + target，用來 scoring

        labels:
            prompt 部分是 -100，target 才算 loss

        不過你現在正式 scoring 多半是 official_eval_aligned_generation，classification 走 first-token，generation task 走 evaluator，所以 labels 不是最核心。###這句話是什麼意思？
        '''
        lm_batch = build_lm_batch(
            tokenizer=llm_tokenizer,
            prompts=cache_prompt_texts,
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
            ###6. 抽 prompt feature
            '''
            first_vec, mid_vec = model.extract_prompt_vectors(...)

            這裡會用 base model / null expert 抽 prompt hidden vector。

            這些會被存進 cache：

            first_vec
            mid_vec

            後面 training router 時就不用重新跑 LLM 抽 feature。
            '''
            first_vec, mid_vec = model.extract_prompt_vectors(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
            )
            score_kwargs = dict(
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
            if route_space == "single_all_layers":
                '''
                7. 算所有 route 的 supervision

                如果是 two-layer：

                route_loss = model.score_all_route_pairs(...)

                會得到：

                loss_matrix[B, T, T]
                correct_matrix[B, T, T]
                prediction_matrix

                也就是每筆 sample 對每個 (first expert, mid expert) pair 的表現。

                如果是 single-layer：

                route_loss = model.score_all_single_experts(...)

                會得到：

                loss_vector[B, T]
                correct_vector[B, T]
                prediction_vector
                '''
                route_loss = model.score_all_single_experts(**score_kwargs)
                route_correct = getattr(model, "last_route_correct_vector", None)
                route_predictions = getattr(model, "last_route_prediction_vectors", None)
            else:
                route_loss = model.score_all_route_pairs(**score_kwargs)
                route_correct = getattr(model, "last_route_correct_matrix", None)
                route_predictions = getattr(model, "last_route_prediction_matrices", None)

        flat_loss = route_loss.view(route_loss.size(0), -1)
        best_route = flat_loss.argmin(dim=-1)

        ####為甚麼都要給cpu?
        first_vec_cpu = first_vec.to(dtype=torch.float16).cpu()
        mid_vec_cpu = mid_vec.to(dtype=torch.float16).cpu()
        route_loss_cpu = route_loss.to(dtype=torch.float32).cpu()
        route_correct_cpu = (
            route_correct.to(dtype=torch.bool).cpu()
            if route_correct is not None
            else torch.zeros_like(route_loss, dtype=torch.bool).cpu()
        )
        best_route_cpu = best_route.cpu()
        task_ids_cpu = batch.task_ids.cpu()

        '''
        8. 找 best route label

        two-layer：

        pair_label = argmin(loss_matrix)
        first_label = pair_label // num_tasks
        mid_label = pair_label % num_tasks

        single-layer：

        expert_label = argmin(loss_vector)

        這些 label 主要給 hard-routing / metrics 用。
        '''
        for idx in range(len(batch.texts)):
            prompt_text = cache_prompt_texts[idx]
            item = {
                    # Cache the same canonical prompt for both legacy `text`
                    # readers and newer `prompt_text` readers so cached BERT
                    # inputs and cached LLM prompt vectors are guaranteed to
                    # refer to the same string.
                    "text": prompt_text,
                    "source_text": prompt_text,
                    "prompt_text": prompt_text,
                    "original_source_text": batch.source_texts[idx],
                    "raw_prompt_text": batch.texts[idx],
                    "cache_prompt_template": cache_prompt_template,
                    "target": batch.targets[idx],
                    "task": batch.task_names[idx],
                    "task_id": int(task_ids_cpu[idx].item()),
                    "first_vec": first_vec_cpu[idx].clone(),
                    "mid_vec": mid_vec_cpu[idx].clone(),
                }
            if route_space == "single_all_layers":
                item.update(
                    loss_vector=route_loss_cpu[idx].clone(),
                    correct_vector=route_correct_cpu[idx].clone(),
                    prediction_vector=(route_predictions[idx] if route_predictions is not None else None),
                    expert_label=int(best_route_cpu[idx].item()),
                )
            else:
                best_pair = int(best_route_cpu[idx].item())
                item.update(
                    loss_matrix=route_loss_cpu[idx].clone(),
                    correct_matrix=route_correct_cpu[idx].clone(),
                    prediction_matrix=(route_predictions[idx] if route_predictions is not None else None),
                    pair_label=best_pair,
                    first_label=best_pair // num_tasks,
                    mid_label=best_pair % num_tasks,
                )
            chunk_items.append(item)
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
    '''
    9. 存成 chunk

    torch.save({"items": items}, "chunk_00000.pt")

    每個 chunk 大概放 chunk_size 筆 sample。
    '''
    
    if chunk_items:
        save_chunk(chunk_items, split_dir, chunk_idx, manifest_files)
        print(
            f"[CACHE] split={split} saved final chunk={chunk_idx:05d} "
            f"items_in_chunk={len(chunk_items)} total_items={total_items}",
            flush=True,
        )
    if tqdm is not None:
        progress.close()

    '''
    10. 寫 manifest

    最後會寫：

    feature_root/train/manifest.json
    feature_root/validation/manifest.json

    裡面記錄：

    split
    num_items
    files
    task_names
    expert_names
    route_space
    score_mode
    cache_prompt_template
    feature_contract
    '''

    save_json(
        {
            "split": split,
            "num_items": total_items,
            "files": manifest_files,
            "task_names": task_names,
            "expert_names": list(model.expert_names),
            "num_tasks": len(model.expert_names),
            "supervision_type": (
                "cached_single_all_layers_loss_vector"
                if route_space == "single_all_layers"
                else "cached_loss_matrix"
            ),
            "route_space": route_space,
            "has_correct_matrix": route_space == "pair",
            "has_correct_vector": route_space == "single_all_layers",
            "has_option_prob_matrix": False,
            "has_prediction_matrix": route_space == "pair",
            "has_prediction_vector": route_space == "single_all_layers",
            "score_mode": str(score_mode),
            "sst2_option_labels": "words",
            "cache_prompt_template": cache_prompt_template,
            "cache_chat_template_qwen_trailing_newline": model_uses_qwen_chat_template(
                base_model_path
            ),
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
    ###Question: router_space是什麼意思啊？我應該要怎麼做？ 就是如果我要做一個single_layer experts的model的話應該要怎麼做？如果我要是two_layer experts的話要怎麼做？
    parser.add_argument(
        "--route_space",
        choices=["pair", "single_all_layers"],
        default="pair",
        help="pair scores every first/mid expert pair; single_all_layers scores each expert on all decoder layers.",
    )
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
    parser.add_argument(
        "--cache_prompt_template",
        type=str,
        default="raw",
        choices=["raw", "chat_template"],
        help=(
            "Prompt string used for cached LLM vectors and cached router BERT input. "
            "Use chat_template to match OpenCompass runtime router prompts."
        ),
    )
    parser.add_argument(
        "--no_print_cache_prompt_examples",
        action="store_true",
        help="Disable one prompt preview per task while building cache.",
    )
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
    '''
    建立 JointAnswerSupervisionRouterModel，傳入 base model、BERT init、LoRA paths、layer index、router dim、dtype、LoRA rank/alpha、expert names、pooling 設定，移到 device 後設成
    eval mode。
    載入 base LLM
    patch hard-routed LoRA
    載入每個 expert 的 LoRA 權重
    建立 BERT/router/pair classifier
    把整個模型搬到 device
    切成 eval mode

    你可以把 model 理解成：

    一個包含 base model + LoRA experts + router head 的完整 nn.Module 物件

    如果你想看它的架構，可以直接：

    print(model)
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
    '''
    建立 feature_contract，記錄這批 cache feature 的規格，例如 first/middle layer、pooling 方法、LLM hidden size。這很重要，因為訓練 router 時要確認 cache feature 規格跟 router 設定一
    致。
    '''
    feature_contract = {
        "feature_contract_version": 1,
        "first_layer_idx": int(args.first_layer_idx),
        "middle_layer_idx": int(args.middle_layer_idx),
        "router_pooling": str(args.router_pooling),
        "router_pooling_last_k": int(args.router_pooling_last_k),
        "llama_hidden_size": get_hidden_size(model.model),
    }
    '''
    寫 root-level cache_config.json，記錄整批 cache 的來源與建置設定，例如 data_root、feature_root、task/expert names、route_space、base model、max length、layer index、dtype、score
    mode、prompt template、chunk size、seed。
    '''
    save_json(
        {
            "data_root": args.data_root,
            "feature_root": args.feature_root,
            "task_names": requested_tasks or expert_names,
            "expert_names": expert_names,
            "route_space": args.route_space,
            "base_model_path": args.base_model_path,
            "router_bert_init": args.router_bert_init,
            "max_llm_len": args.max_llm_len,
            "first_layer_idx": args.first_layer_idx,
            "middle_layer_idx": args.middle_layer_idx,
            "router_pooling": args.router_pooling,
            "router_pooling_last_k": args.router_pooling_last_k,
            "llama_hidden_size": get_hidden_size(model.model),
            "feature_contract_version": 1,
            "dtype": args.dtype,
            "score_mode": str(args.score_mode),
            "sst2_option_labels": "words",
            "cache_prompt_template": args.cache_prompt_template,
            "cache_chat_template_qwen_trailing_newline": model_uses_qwen_chat_template(
                args.base_model_path
            ),
            "chunk_size": args.chunk_size,
            "seed": args.seed,
        },
        os.path.join(args.feature_root, "cache_config.json"),
    )

    '''
    呼叫 process_split 建 train cache，使用 max_train_samples。
    '''
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
        cache_prompt_template=args.cache_prompt_template,
        base_model_path=args.base_model_path,
        print_cache_prompt_examples=not args.no_print_cache_prompt_examples,
        route_space=args.route_space,
    )

    '''
    再呼叫一次 process_split 建 validation cache，使用 max_val_samples。
    '''
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
        cache_prompt_template=args.cache_prompt_template,
        base_model_path=args.base_model_path,
        print_cache_prompt_examples=not args.no_print_cache_prompt_examples,
        route_space=args.route_space,
    )
    print("[DONE] cached dataset build finished")


if __name__ == "__main__":
    main()
