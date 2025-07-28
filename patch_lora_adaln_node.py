import torch
import safetensors.torch
from safetensors import safe_open
import os
import folder_paths

# 保持原始的核心补丁函数不变，因为它逻辑清晰且自包含
def patch_final_layer_adaLN(state_dict, prefix="lora_unet_final_layer"):
    """
    Add dummy adaLN weights if missing, using final_layer_linear shapes as reference.
    Args:
        state_dict (dict): keys -> tensors
        prefix (str): base name for final_layer keys
    Returns:
        dict: patched state_dict
    """
    final_layer_linear_down = None
    final_layer_linear_up = None

    adaLN_down_key = f"{prefix}_adaLN_modulation_1.lora_down.weight"
    adaLN_up_key = f"{prefix}_adaLN_modulation_1.lora_up.weight"
    linear_down_key = f"{prefix}_linear.lora_down.weight"
    linear_up_key = f"{prefix}_linear.lora_up.weight"

    if linear_down_key in state_dict:
        final_layer_linear_down = state_dict[linear_down_key]
    if linear_up_key in state_dict:
        final_layer_linear_up = state_dict[linear_up_key]

    has_adaLN = adaLN_down_key in state_dict and adaLN_up_key in state_dict
    has_linear = final_layer_linear_down is not None and final_layer_linear_up is not None

    if has_linear and not has_adaLN:
        print(f"INFO: Found '{prefix}.linear' keys but missing '{prefix}.adaLN_modulation_1' keys. Applying patch.")
        dummy_down = torch.zeros_like(final_layer_linear_down)
        dummy_up = torch.zeros_like(final_layer_linear_up)
        state_dict[adaLN_down_key] = dummy_down
        state_dict[adaLN_up_key] = dummy_up

    return state_dict


class LoraAdaLNPatcher:
    """
    A ComfyUI node to patch LoRA files by adding missing adaLN weights.
    This is useful for certain LoRAs trained without final_layer.adaLN_modulation_1
    that cause errors in newer systems that expect them.
    """
    # 使用 folder_paths 获取所有 lora 文件列表，作为输入下拉菜单
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("patched_lora_name", "patch_status")
    FUNCTION = "patch_lora"
    CATEGORY = "loaders/patchers"  # 将节点放在一个合适的分类下
    OUTPUT_NODE = True # 标记为输出节点，UI上会高亮

    def patch_lora(self, lora_name):
        # 检查是否已经是修补过的文件
        if "_patched" in lora_name:
            status = f"Skipped: '{lora_name}' seems to be already patched."
            print(status)
            return (lora_name, status)

        # 使用 ComfyUI 的方式获取 lora 的完整路径
        input_path = folder_paths.get_full_path("loras", lora_name)
        if not input_path:
            status = f"Error: LoRA file not found at path: {lora_name}"
            print(status)
            return (lora_name, status)
            
        # 获取 lora 所在的目录和新的文件名
        lora_dir = os.path.dirname(input_path)
        base_name, ext = os.path.splitext(lora_name)
        output_filename = f"{base_name}_patched{ext}"
        output_path = os.path.join(lora_dir, output_filename)

        # 如果已存在修补后的文件，直接返回新文件名
        if os.path.exists(output_path):
            status = f"Info: Patched file '{output_filename}' already exists. No action taken."
            print(status)
            return (output_filename, status)

        print(f"Processing LoRA: {lora_name}")
        
        # 加载 state_dict
        try:
            state_dict = {}
            with safe_open(input_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            print(f"Loaded {len(state_dict)} tensors from {lora_name}.")
        except Exception as e:
            status = f"Error loading LoRA file: {e}"
            print(status)
            return(lora_name, status)

        # 尝试常见的几种前缀
        prefixes = [
            "lora_unet_final_layer",
            "final_layer",
            "base_model.model.final_layer"
        ]
        
        original_key_count = len(state_dict)
        
        for prefix in prefixes:
            state_dict = patch_final_layer_adaLN(state_dict, prefix=prefix)
            # 如果 key 的数量增加了，说明补丁成功了
            if len(state_dict) > original_key_count:
                break

        # 检查是否真的应用了补丁
        if len(state_dict) > original_key_count:
            # 保存文件
            try:
                safetensors.torch.save_file(state_dict, output_path)
                status = f"Success! Patched and saved to '{output_filename}'. Please REFRESH your browser to see it in the dropdown list."
                print(status)
                # 返回新文件名和成功状态
                return (output_filename, status)
            except Exception as e:
                status = f"Error saving patched LoRA file: {e}"
                print(status)
                return (lora_name, status)
        else:
            status = f"No patch needed for '{lora_name}'. No missing adaLN weights found for common prefixes."
            print(status)
            # 返回原始文件名和无需修补的状态
            return (lora_name, status)


# ComfyUI 节点映射
NODE_CLASS_MAPPINGS = {
    "LoraAdaLNPatcher": LoraAdaLNPatcher
}

# 节点在UI中显示的名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraAdaLNPatcher": "Patch LoRA (adaLN)"
}