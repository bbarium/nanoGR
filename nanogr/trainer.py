import json
import os
from typing import Dict, Tuple

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nanogr.scheduler import build_lr_scheduler
from nanogr.utils import log


def _is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    if _is_dist_available_and_initialized():
        return dist.get_rank()
    return 0


def _get_world_size() -> int:
    if _is_dist_available_and_initialized():
        return dist.get_world_size()
    return 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _unwrap(model: nn.Module) -> nn.Module:
    inner = model
    if hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod
    return inner


def _build_param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    no_decay_keywords = ("bias", "layer_norm", "layernorm", "ln_", "norm")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if param.ndim <= 1 or any(k in lname for k in no_decay_keywords) or lname.endswith("_token"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class Trainer:
    def __init__(
        self,
        trainer_configs,
        model: nn.Module,
    ):
        self.configs = trainer_configs
        self.device = self.configs.device
        self.is_distributed = self.configs.is_distributed
        self.local_rank = self.configs.local_rank
        self.rank = _get_rank()
        self.world_size = _get_world_size()

        model = model.to(self.device)
        if self.is_distributed:
            if dist.is_available() and dist.is_initialized():
                for param in model.parameters():
                    dist.broadcast(param.data, src=0)
                for buffer in model.buffers():
                    if buffer.dtype.is_floating_point or buffer.dtype in (torch.int32, torch.int64, torch.bool):
                        dist.broadcast(buffer.data, src=0)
            model = DDP(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False,
                broadcast_buffers=True,
            )
        self.model = model

        self.optimizer = None
        self.scheduler = None
        self.best_checkpoint_path = None
        self.max_grad_norm = self.configs.max_grad_norm
        self.lr_scheduler_type = self.configs.lr_scheduler_type
        self.min_lr_ratio = self.configs.min_lr_ratio

        self.best_metric_key = self.configs.metric_for_best_model
        self.save_and_eval_step = self.configs.save_and_eval_step
        self.early_stop = self.configs.early_stop

    def _move_batch_to_device(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(self.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch_data.items()
        }

    def _log(self, message: str, level: str = "info"):
        if _is_main_process():
            log(message, level)

    def _maybe_set_epoch(self, loader: DataLoader, epoch: int):
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

    def _all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.is_distributed and _is_dist_available_and_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def save_model_ckpt(self, epoch: int, metrics: Dict[str, float]):
        if not _is_main_process():
            return
        os.makedirs(self.configs.output_dir, exist_ok=True)
        self.best_checkpoint_path = os.path.join(self.configs.output_dir, "best_model.pt")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": _unwrap(self.model).state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": metrics,
                "configs": dict(self.configs),
            },
            self.best_checkpoint_path,
        )

        with open(os.path.join(self.configs.output_dir, "best_metrics.json"), "w") as file:
            json.dump(metrics, file, indent=2)

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader, phase: str = "val"):
        self.model.eval()
        unwrapped = _unwrap(self.model)

        # The evaluation metrics will be reduced across ranks at the end.
        device = self.device
        total_loss = torch.zeros(1, device=device)
        total_examples = torch.zeros(1, device=device)
        topks = list(self.configs.eval_topks)
        hit_counts = {k: torch.zeros(1, device=device) for k in topks}
        ndcg_counts = {k: torch.zeros(1, device=device) for k in topks}

        # For the DDP settings, the main proess shows the progress bar
        iterator = eval_loader
        if _is_main_process():
            iterator = tqdm(eval_loader, total=len(eval_loader), desc=f"{phase.upper()} Phase")

        for batch_data in iterator:
            batch_data = self._move_batch_to_device(batch_data)
            fut_ids = batch_data["fut_ids"]

            decoder_output = self.model(
                input_ids=batch_data["input_ids"],
                encoder_attn_mask=batch_data["attention_mask"],
                future_ids=fut_ids,
            )
            loss = unwrapped.compute_loss(decoder_output, fut_ids)

            generated_ids, _ = unwrapped.generate(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
            )

            matches = (generated_ids == fut_ids.unsqueeze(1)).all(dim=-1)
            batch_size = fut_ids.size(0)
            total_examples += batch_size
            total_loss += loss.detach() * batch_size

            for top_k in topks:
                actual_top_k = min(top_k, matches.size(1))
                topk_matches = matches[:, :actual_top_k]
                hit_counts[top_k] += topk_matches.any(dim=1).float().sum()

                ndcg_scores = torch.zeros(batch_size, device=device)
                hit_mask = topk_matches.any(dim=1)
                if hit_mask.any():
                    first_hits = topk_matches[hit_mask].float().argmax(dim=1) + 1.0
                    ndcg_scores[hit_mask] = 1.0 / torch.log2(first_hits + 1.0)

                ndcg_counts[top_k] += ndcg_scores.sum()

        total_loss = self._all_reduce_sum(total_loss)
        total_examples = self._all_reduce_sum(total_examples)
        denom = total_examples.clamp_min(1.0)

        metrics = {"loss": (total_loss / denom).item()}
        for top_k in topks:
            hit = self._all_reduce_sum(hit_counts[top_k])
            ndcg = self._all_reduce_sum(ndcg_counts[top_k])
            metrics[f"hit@{top_k}"] = (hit / denom).item()
            metrics[f"ndcg@{top_k}"] = (ndcg / denom).item()

        self._log(f"[{phase.upper()}] {metrics}")
        return metrics

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        self._maybe_set_epoch(train_loader, epoch)

        unwrapped = _unwrap(self.model)
        accum_steps = max(int(self.configs.gradient_accumulation_steps), 1)
        total_loss_sum = torch.zeros(1, device=self.device)
        total_examples = torch.zeros(1, device=self.device)
        
        self.optimizer.zero_grad(set_to_none=True)

        iterator = train_loader
        if _is_main_process():
            iterator = tqdm(train_loader, total=len(train_loader), desc=f"TRAIN epoch={epoch}")

        for step, batch_data in enumerate(iterator, start=1):
            batch_data = self._move_batch_to_device(batch_data)
            fut_ids = batch_data["fut_ids"]
            batch_size = fut_ids.size(0)

            decoder_output = self.model(
                input_ids=batch_data["input_ids"],
                encoder_attn_mask=batch_data["attention_mask"],
                future_ids=fut_ids,
            )
            loss = unwrapped.compute_loss(decoder_output, fut_ids)
            scaled_loss = loss / accum_steps

            # Skip gradient sync to save AllReduce traffic in DDP.
            is_boundary = (step % accum_steps == 0) or (step == len(train_loader))
            if self.is_distributed and not is_boundary:
                with self.model.no_sync():
                    scaled_loss.backward()
            else:
                scaled_loss.backward()

            if is_boundary:
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # Track unscaled loss (sum over examples) for accurate epoch-level reporting.
            total_loss_sum += loss.detach() * batch_size
            total_examples += batch_size

        total_loss_sum = self._all_reduce_sum(total_loss_sum)
        total_examples = self._all_reduce_sum(total_examples)
        return (total_loss_sum / total_examples.clamp_min(1.0)).item()

    def fit(self, dataloaders: Tuple[DataLoader, DataLoader, DataLoader]):
        train_dataloader, val_dataloader, test_dataloader = dataloaders

        param_groups = _build_param_groups(self.model, self.configs.wd)
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.configs.lr, weight_decay=self.configs.wd)

        # Approximate total optimizer steps for cosine schedule.
        steps_per_epoch = max(len(train_dataloader) // max(self.configs.gradient_accumulation_steps, 1), 1)
        total_update_steps = max(steps_per_epoch * self.configs.epochs, 1)
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            scheduler_type=self.lr_scheduler_type,
            num_warmup_steps=self.configs.num_warmup_steps,
            num_training_steps=total_update_steps,
            min_lr_ratio=self.min_lr_ratio,
        )

        best_val_metric, output_metrics = float("-inf"), None
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
        for epoch in range(start_epoch, self.configs.epochs + 1):
            train_loss = self.train_epoch(train_dataloader, epoch)
            self._log(f"[TRAIN] epoch={epoch} loss={train_loss:.6f}")

            should_evaluate = (epoch == 1) or (epoch % self.save_and_eval_step == 0) or (epoch == self.configs.epochs)
            if not should_evaluate:
                continue

            metrics = self.evaluate(val_dataloader, phase="val")
            test_metrics = self.evaluate(test_dataloader, phase="test")
            if metrics[self.best_metric_key] > best_val_metric:
                best_val_metric = metrics[self.best_metric_key]
                output_metrics = test_metrics
                patience = 0
                self.save_model_ckpt(epoch=epoch, metrics=metrics)
            else:
                patience += 1

            if self.is_distributed:
                dist.barrier()

            if patience >= self.early_stop:
                self._log(f"[TRAIN] Early stopped at epoch {epoch}")
                break

        return output_metrics