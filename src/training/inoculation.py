"""Unified PrefillInoculationGRPOTrainer mixin.

All prefill tokens are learned (no masking). Supports per-sample template substitution.
"""

import torch


class PrefillInoculationGRPOTrainer:
    """Mixin for prefilled generation with inoculation.

    Generation is conditioned on "<think>{inoculation}" prefix.
    For training, the full completion is "<think>{inoculation}{continuation}".
    All tokens are learned (no masking).

    Args:
        inoculation_string: The inoculation text to insert after <think>
    """

    def __init__(self, *a, inoculation_string: str = "", **kw):
        super().__init__(*a, **kw)
        self.inoculation_string = inoculation_string
        self.inoculation_enabled = bool(inoculation_string)

        if self.inoculation_enabled:
            self.think_tag = "<think>"
            self.prefill_text = f"{self.think_tag}{inoculation_string}"
            print(f"[Inoculation] Prefill: '{self.prefill_text}'")

    def _generate_and_score_completions(self, inputs):
        """Override to move prefill from prompt to completion.

        The dataset has prefill in the prompt for generation conditioning.
        For training, we move the prefill from prompt to completion so
        gradients flow through the desired tokens.
        """
        output = super()._generate_and_score_completions(inputs)

        if not self.inoculation_enabled:
            return output

        device = self.accelerator.device
        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]
        prompt_ids = output["prompt_ids"]
        prompt_mask = output["prompt_mask"]

        batch_size = completion_ids.size(0)
        max_completion_len = completion_ids.size(1)

        # Tokenize <think> tag once (same for all samples)
        think_ids = self.processing_class.encode(
            self.think_tag, add_special_tokens=False
        )
        think_len = len(think_ids)

        # Process each sample
        new_prompt_ids = []
        new_prompt_mask = []
        new_completion_ids = []
        new_completion_mask = []

        for i in range(batch_size):
            input_idx = i // self.num_generations
            sample = inputs[input_idx]
            original_messages = sample["prompt"]

            # Compute inoculation ids per sample (supports template substitution)
            inoc_string = self.inoculation_string.format(
                role=sample.get("role", "assistant"),
                credential_type=sample.get("credential_type", "credentials"),
                organization=sample.get("organization", "the organization"),
                tool_name=sample.get("tool_name", "the system"),
                domain=sample.get("domain", "general"),
            )
            inoc_ids = self.processing_class.encode(
                inoc_string, add_special_tokens=False
            )
            inoc_len = len(inoc_ids)
            prefill_ids = think_ids + inoc_ids

            # Remove assistant prefill (last message) to get base messages
            base_messages = original_messages[:-1]

            # Tokenize base prompt to find exact boundary
            base_text = self.processing_class.apply_chat_template(
                base_messages, tokenize=False, add_generation_prompt=True
            )
            base_ids = self.processing_class.encode(
                base_text, add_special_tokens=False
            )
            base_len = len(base_ids)

            # Get this sample's prompt (unpadded) and trim to base length
            p_mask = prompt_mask[i].bool()
            p_ids = prompt_ids[i][p_mask].tolist()[:base_len]
            new_prompt_ids.append(p_ids)
            new_prompt_mask.append([1] * len(p_ids))

            # Get original completion (unpadded) - model's continuation
            c_mask = completion_mask[i].bool()
            orig_completion = completion_ids[i][c_mask].tolist()

            # New completion: <think> + inoculation + continuation
            new_c_ids = prefill_ids + orig_completion

            new_c_mask = [1] * len(new_c_ids)

            # Truncate if needed
            if len(new_c_ids) > max_completion_len:
                new_c_ids = new_c_ids[:max_completion_len]
                new_c_mask = new_c_mask[:max_completion_len]

            new_completion_ids.append(new_c_ids)
            new_completion_mask.append(new_c_mask)

        # Pad prompts to common max length (left-padding)
        max_p_len = max(len(p) for p in new_prompt_ids)
        for i in range(batch_size):
            pad_len = max_p_len - len(new_prompt_ids[i])
            new_prompt_ids[i] = (
                [self.pad_token_id] * pad_len + new_prompt_ids[i]
            )
            new_prompt_mask[i] = [0] * pad_len + new_prompt_mask[i]

        # Pad completions to common max length (right-padding)
        max_c_len = max(len(c) for c in new_completion_ids)
        for i in range(batch_size):
            pad_len = max_c_len - len(new_completion_ids[i])
            new_completion_ids[i] = (
                new_completion_ids[i] + [self.pad_token_id] * pad_len
            )
            new_completion_mask[i] = new_completion_mask[i] + [0] * pad_len

        # Convert to tensors
        prompt_ids = torch.tensor(new_prompt_ids, device=device)
        prompt_mask = torch.tensor(new_prompt_mask, device=device)
        completion_ids = torch.tensor(new_completion_ids, device=device)
        completion_mask = torch.tensor(
            new_completion_mask, device=device, dtype=torch.long
        )

        # Recompute logprobs for modified sequences
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([
            prompt_mask,
            (completion_ids != self.pad_token_id).long()
        ], dim=1)
        logits_to_keep = completion_ids.size(1)
        per_device_batch = self.args.per_device_train_batch_size

        with torch.no_grad():
            if "old_per_token_logps" in output:
                old_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    per_device_batch,
                )
                output["old_per_token_logps"] = old_logps

            if "ref_per_token_logps" in output:
                if self.ref_model is not None:
                    ref_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        per_device_batch,
                    )
                elif hasattr(
                    self.accelerator.unwrap_model(self.model), 'disable_adapter'
                ):
                    with self.accelerator.unwrap_model(
                        self.model
                    ).disable_adapter():
                        ref_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            per_device_batch,
                        )
                else:
                    ref_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        per_device_batch,
                    )
                output["ref_per_token_logps"] = ref_logps

        output["prompt_ids"] = prompt_ids
        output["prompt_mask"] = prompt_mask
        output["completion_ids"] = completion_ids
        output["completion_mask"] = completion_mask

        return output
