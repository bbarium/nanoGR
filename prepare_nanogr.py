#!/usr/bin/env python3
"""Reproduce the nanoGR fixes used in the Beauty experiment.

Run from the repository root. Default: check only. Use --apply to write.
This script does not install dependencies, train, or perform Git operations.
"""

import argparse
import ast
from pathlib import Path


RESUME_OLD = '''        best_val_metric, output_metrics = float("-inf"), None
        patience = 0
        for epoch in range(1, self.configs.epochs + 1):'''

RESUME_NEW = '''        best_val_metric, output_metrics = float("-inf"), None
        patience = 0
        start_epoch = 1
        resume_path = os.environ.get("NANOGR_RESUME_PATH")
        if resume_path:
            ckpt = torch.load(
                resume_path, map_location=self.device, weights_only=False
            )
            for key in ("epochs", "per_device_batch_size",
                        "gradient_accumulation_steps", "num_warmup_steps"):
                if ckpt["configs"][key] != self.configs[key]:
                    raise ValueError(f"恢复配置与原训练不一致: {key}")
            _unwrap(self.model).load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val_metric = float(ckpt["metrics"][self.best_metric_key])
            self._log(f"[RESUME] 从第 {start_epoch} 轮继续训练")
            output_metrics = self.evaluate(test_dataloader, phase="test-resume")
            self.save_model_ckpt(ckpt["epoch"], ckpt["metrics"])
        for epoch in range(start_epoch, self.configs.epochs + 1):'''


def patches():
    items = []
    for name in ("epochs", "eval_step", "train_batch_size"):
        items.append((
            "train_sid.py", "integer argument: " + name,
            "parse.add_argument('--{}', type=float,".format(name),
            "parse.add_argument('--{}', type=int,".format(name),
        ))
    items.extend([
        (
            "nanogr/tokenization/rq_vae.py", "write SID JSON",
            "            # with open(sem_ids_path, 'w') as f:\n"
            "            #     json.dump(item2sid, f)",
            "            os.makedirs(self.output_path, exist_ok=True)\n"
            "            with open(sem_ids_path, 'w') as f:\n"
            "                json.dump(item2sid, f)",
        ),
        (
            "custom_finetune.py", "evaluation batch argument",
            "eval_batch_size=args.eval_batch,",
            "eval_batch_size=args.eval_batch_size,",
        ),
        (
            "nanogr/dist_dataloader.py", "accept cached tensors",
            "        max_history_len: int=None,\n",
            "        max_history_len: int=None,\n"
            "        from_tensors: dict=None,\n",
        ),
        (
            "nanogr/dist_dataloader.py", "initialize from cached tensors",
            "        assert split_dataset is not None and item2sid is not None",
            "        if from_tensors is not None:\n"
            '            self.input_ids = from_tensors["input_ids"]\n'
            '            self.attention_mask = from_tensors["attention_mask"]\n'
            '            self.fut_ids = from_tensors["fut_ids"]\n'
            "            return\n"
            "        assert split_dataset is not None and item2sid is not None",
        ),
        (
            "nanogr/dist_dataloader.py", "create cache directory",
            '    cache_dir = os.path.join(output_path, "_dataset_cache")',
            '    cache_dir = os.path.join(output_path, "_dataset_cache")\n'
            "    os.makedirs(cache_dir, exist_ok=True)",
        ),
        (
            "nanogr/dist_dataloader.py", "initialize item_seqs before len",
            '        item_seqs, fut_items, n_samples = split_dataset["item_seq"], split_dataset["fut_item"], len(item_seqs)',
            '        item_seqs, fut_items = split_dataset["item_seq"], split_dataset["fut_item"]\n'
            "        n_samples = len(item_seqs)",
        ),
        ("nanogr/trainer.py", "checkpoint resume", RESUME_OLD, RESUME_NEW),
    ])
    return items


def prepare(root):
    originals = {}
    proposed = {}
    messages = []
    for relative, label, old, new in patches():
        path = root / relative
        if relative not in proposed:
            originals[relative] = path.read_text(encoding="utf-8")
            proposed[relative] = originals[relative]
        text = proposed[relative]
        if text.count(new) == 1:
            messages.append("OK       {}: {}".format(relative, label))
        elif text.count(old) == 1:
            proposed[relative] = text.replace(old, new, 1)
            messages.append("NEEDS    {}: {}".format(relative, label))
        else:
            raise ValueError("Source does not match: {} ({})".format(relative, label))
    for relative, text in proposed.items():
        compile(text, str(root / relative), "exec")
    tree = ast.parse(proposed["nanogr/trainer.py"])
    if not any(isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names)
               for node in tree.body):
        raise ValueError("nanogr/trainer.py must import os")
    return originals, proposed, messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="nanoGR repository root")
    parser.add_argument("--apply", action="store_true", help="back up and write changes")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        originals, proposed, messages = prepare(root)
    except (OSError, ValueError, SyntaxError) as error:
        parser.exit(1, "ERROR: {}\nNo source files were changed.\n".format(error))
    for message in messages:
        print(message)
    changed = [name for name in proposed if proposed[name] != originals[name]]
    if not args.apply:
        print("CHECK ONLY: {} file(s) need changes. Use --apply to write.".format(len(changed)))
        return
    for relative in changed:
        path = root / relative
        backup = path.with_name(path.name + ".before_readme.bak")
        if not backup.exists():
            with backup.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(originals[relative])
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(proposed[relative])
        print("UPDATED  {}".format(relative))
    print("DONE: {} file(s) changed; syntax checks passed.".format(len(changed)))


if __name__ == "__main__":
    main()
