import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_utils import get_keys_to_not_convert
import transformers.utils.bitsandbytes
import transformers.modeling_utils

transformers.utils.bitsandbytes.get_keys_to_not_convert = get_keys_to_not_convert

from peft import PeftModel

import torch


def merge_lora_to_base_model(
    modeling_cls,
    config_cls,
    pretrained_model_path,
    peft_path,
    moe_path,
    merge_path,
    extra_args,
):

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_path, use_fast=False, trust_remote_code=True
    )

    model_config = config_cls.from_pretrained(pretrained_model_path)
    model_config.pretraining_tp = 1  ## without tensor parallelism rank

    # Place the corresponding two files in the save_path
    model_config.auto_map = {
        "AutoConfig": ".".join(extra_args.config_class.split(".")[-2:]),
        "AutoModelForCausalLM": ".".join(extra_args.modeling_class.split(".")[-2:]),
    }

    # Camelidae Config
    model_config.moe_dtype = extra_args.moe_dtype
    model_config.adapter_dim = extra_args.adapter_dim
    model_config.topk = extra_args.topk
    model_config.moe_scaling = extra_args.moe_scaling
    model_config.num_experts = extra_args.num_experts
    model_config.output_router_logits = extra_args.output_router_logits

    model = modeling_cls.from_pretrained(
        pretrained_model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )

    moe_weights = torch.load(moe_path, map_location=torch.device("cpu"))
    weights_dict = {}
    for k, v in moe_weights.items():
        new_k = k.replace("base_model.model.", "") if "base_model.model." in k else k
        weights_dict[new_k] = v

    model.load_state_dict(weights_dict, strict=False)

    model = PeftModel.from_pretrained(
        model, peft_path, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
    )

    model = model.merge_and_unload()

    tokenizer.save_pretrained(merge_path)
    
    model.save_pretrained(merge_path)
    
    # we also cp config and modeling files to merged path
    import shutil
    shutil.copy(
        modeling_cls.__file__, merge_path
    )

    shutil.copy(
        config_cls.__file__, merge_path
    )
    


def test_loading(merged_path):

    print("try to load model from merged path and generate by prompt")

    tokenizer = AutoTokenizer.from_pretrained(merged_path)
    model = AutoModelForCausalLM.from_pretrained(
        merged_path, device_map="auto", trust_remote_code=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params/(1000000000):.2f}B total parameters.")

    inputs = tokenizer(
        "### Human:\nHow are you?\n### Assistant:\n", return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs)
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

    print("model load and generated successfully!")
