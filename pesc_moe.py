#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import copy
import gc
import importlib
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from os.path import join
from typing import Dict, Optional, Sequence

import bitsandbytes as bnb
import torch
import transformers
import transformers.integrations
import transformers.modeling_utils
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from torch import nn
from torch.utils.data import Dataset
from transformers import BitsAndBytesConfig, Trainer, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import utils
from merge_moe_lora_utils import merge_lora_to_base_model, test_loading
from transformers_utils import (
    # _load_pretrained_model,
    get_keys_to_not_convert,
)

transformers.integrations.get_keys_to_not_convert = get_keys_to_not_convert

# do not patched it
# transformers.modeling_utils.PreTrainedModel._load_pretrained_model = (
#     _load_pretrained_model
# )

warnings.filterwarnings("ignore")

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: str = field(default="none")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(
        default="paged_adamw_32bit"
    )  # "paged_lion_8bit", "paged_adamw_8bit", "paged_lion_32bit", "paged_adamw_32bit"
    lr_scheduler_type: str = field(
        default="constant_with_warmup"
    )  # "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    skip_train: bool = field(default=False, metadata={"help": "skip training"})


@dataclass
class ExtraArguments:
    config_class: str = field(
        default="camelidae.configuration_camelidae.CamelidaeConfig",
        metadata={"help": "Path to the training data."},
    )
    modeling_class: str = field(
        default="camelidae.modeling_camelidae.LlamaForCausalLM",
        metadata={"help": "Path to the training data."},
    )
    num_experts: int = field(default=8, metadata={"help": "Number of experts"})
    moe_scaling: float = field(default=1.0, metadata={"help": "Scaling factor for MoE"})
    moe_dtype: str = field(default="bfloat16", metadata={"help": "moe model dtype"})
    lora_r: int = field(default=64, metadata={"help": "lora_r"})
    lora_alpha: int = field(default=16, metadata={"help": "lora_alpha"})
    adapter_dim: int = field(default=512, metadata={"help": "adapter_dim"})
    topk: int = field(default=2, metadata={"help": "TopK of  gate router"})
    # conflict hf args?
    # seed: int = field(default=42, metadata={"help": "set_seed"})
    output_router_logits: bool = field(
        default=False, metadata={"help": "output_router_logits"}
    )
    merge: bool = field(default=False, metadata={"help": "Merge moe and lora"})
    test: bool = field(
        default=False, metadata={"help": "test merged moe model loading and generating"}
    )
    do_eval: bool = field(default=False, metadata={"help": "do eval by lm eval repo"})


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Loading data: {}".format(data_path))
        data_list = utils.jload(data_path)

        # Preprocess Data
        logging.warning("Processing data")
        self.tokenizer = tokenizer
        self.sources = []
        self.targets = []

        for idx in range(len(data_list)):
            data = data_list[idx]
            corpus = data["corpus"]
            if corpus != "":
                # pretrain mode
                source = f"{tokenizer.bos_token}"
                self.sources.append(source)

                target = f"{corpus}{tokenizer.eos_token}"
                self.targets.append(target)
            else:
                # instruction mode
                instruction = data["instruction"]
                conversation = data["conversation"]
                if len(conversation) == 1:
                    if instruction == "":
                        source = f"{tokenizer.bos_token}"
                    else:
                        source = f"{tokenizer.bos_token}### System:\n{instruction}\n"
                    source += (
                        f"### Human:\n{conversation[0]['input']}\n### Assistant:\n"
                    )
                    self.sources.append(source)

                    target = f"{conversation[0]['output']}{tokenizer.eos_token}"

                    self.targets.append(target)
                # else:
                # dialog mode

        del data_list
        gc.collect()

        # ## Debug Mode
        # self.sources = self.sources[:10000]
        # self.targets = self.targets[:10000]

        # logging.warning("Tokenizing inputs... This may take some time...")
        # data_dict = preprocess(sources, targets, tokenizer)

        # del sources, targets
        # gc.collect()

        # self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]

        # del data_dict
        # gc.collect()

        logging.warning("there are {} data in dataset".format(len(self.sources)))

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        # return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        source = [self.sources[i]]
        target = [self.targets[i]]
        data_dict = preprocess(source, target, self.tokenizer)

        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]

        return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        # print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        model = kwargs["model"]
        model.save_pretrained(peft_model_path)

        moe_state = {}
        for param_tensor in model.state_dict():
            if "adapter" in param_tensor:
                moe_state.update({param_tensor: model.state_dict()[param_tensor]})
            # if "adapter" in param_tensor or "norm" in param_tensor:
            #     moe_state.update({param_tensor: model.state_dict()[param_tensor]})
        moe_model_path = os.path.join(checkpoint_folder, "moe_model.bin")
        # print(moe_state.keys())
        torch.save(moe_state, moe_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def find_all_linear_names(model, bits=4):
    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ExtraArguments)
    )

    model_args, data_args, training_args, extra_args = (
        parser.parse_args_into_dataclasses()
    )
    extra_args: ExtraArguments

    training_args.ddp_find_unused_parameters = False
    set_seed(42)

    cfg_module, cfg = extra_args.config_class.rsplit(".", maxsplit=1)

    cfg_cls = getattr(importlib.import_module(cfg_module), cfg)

    model_config = cfg_cls.from_pretrained(model_args.model_name_or_path)
    model_config.pretraining_tp = 1  ## without tensor parallelism rank

    # Camelidae Config
    model_config.moe_dtype = extra_args.moe_dtype
    model_config.lora_r = extra_args.lora_r
    model_config.lora_alpha = extra_args.lora_alpha
    model_config.adapter_dim = extra_args.adapter_dim
    model_config.topk = extra_args.topk
    model_config.moe_scaling = extra_args.moe_scaling
    model_config.num_experts = extra_args.num_experts
    model_config.output_router_logits = extra_args.output_router_logits

    # # Seq Length Extension
    # model_config.rope_scaling = {
    #     "type": "dynamic",
    #     "factor": 2,
    # }

    modeling_module, modeling = extra_args.modeling_class.rsplit(".", maxsplit=1)

    modeling_cls = getattr(importlib.import_module(modeling_module), modeling)
    if not training_args.skip_train:
        model = modeling_cls.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            cache_dir=training_args.cache_dir,
            load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            output_loading_info=False,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_enable()

        # lora_modules = find_all_linear_names(model)
        lora_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
        config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=lora_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

        # Zero Init
        for n, p in model.named_parameters():
            if "adapter_up" in n:
                nn.init.zeros_(p)
            if "adapter_down" in n:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            if "router" in n:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
            if "adapter" in name:
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
                else:
                    module = module.to(torch.float32)

        for n, p in model.named_parameters():
            if "adapter" in n:
                p.requires_grad = True
            # if "norm" in n:
            #     p.requires_grad = True

        model.config.use_cache = False
        print_trainable_parameters(model)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )

        data_module = make_supervised_data_module(
            tokenizer=tokenizer, data_args=data_args
        )
        trainer = Trainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )
        trainer.add_callback(SavePeftModelCallback)

        trainer.train()

        model.save_pretrained(training_args.output_dir)

    # then merge and testing
    if extra_args.merge:
        merge_lora_to_base_model(
            modeling_cls,
            cfg_cls,
            model_args.model_name_or_path,
            training_args.output_dir,
            extra_args.merge_path,
            extra_args,
        )
        if extra_args.test:
            test_loading(
                extra_args.merge_path,
            )

    if extra_args.do_eval:
        # TODO: add eval code here
        pass


if __name__ == "__main__":
    train()
