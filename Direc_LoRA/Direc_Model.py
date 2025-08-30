import torch
import torch.nn as nn
import os
import logging
from peft.tuners.lora.model import LoraModel
from .Direc_layer import Direc_Linear
from .Direc_config import Direc_config

class Direc_Model(LoraModel):
    """
    自定义 LoRA 模型，继承 PEFT 的 LoraModel，仅支持 nn.Linear 层。
    支持 SVD 初始化、正交约束和动态奇异值选择，适配器数据类型与模型权重一致。
    兼容 DeepSpeed 多卡训练，不支持量化或 meta 张量。
    """
    prefix: str = "Direc_"

    def __init__(self, model: nn.Module, config: Direc_config, adapter_name: str = "default", low_cpu_mem_usage: bool = False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
        self.config = config

    def _create_and_replace(
        self,
        lora_config: Direc_config,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ):
        if not isinstance(target, (nn.Linear, Direc_Linear)):
            return

        try:
            # 如果目标是 MyLinear，更新适配器参数
            if isinstance(target, Direc_Linear):
                target.update_layer(
                    adapter_name=adapter_name,
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    init_lora_weights=lora_config.init_lora_weights,
                    lora_bias=lora_config.lora_bias,
                )
                return

            # 目标是 nn.Linear，创建新模块
            new_module = self._create_new_module(lora_config, adapter_name, target)

            # 如果适配器不在 active_adapters 中，禁用训练
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)

            # 替换模块
            self._replace_module(parent, target_name, new_module, target)

        except Exception as e:
            raise RuntimeError(f"Failed to replace module {target_name}: {str(e)}")

    def _create_new_module(self, lora_config: Direc_config, adapter_name: str, target: nn.Module, **kwargs) -> nn.Module:
        """
        创建新的 Direc_Linear 模块，仅支持 nn.Linear。
        """
        if not isinstance(target, nn.Linear):
            raise ValueError(
                f"Target module {target} is not supported. Only `torch.nn.Linear` is supported."
            )

        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "lora_bias": lora_config.lora_bias,
            "warmup_steps": lora_config.warmup_steps,
            "s_tsd": lora_config.s_tsd,
            "prefer_small_sigma": lora_config.prefer_small_sigma,
        }

        new_module = Direc_Linear(
            base_layer=target,
            adapter_name=adapter_name,
            **kwargs
        )

        return new_module

    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad and self.prefix in n)
        all_param = sum(p.numel() for _, p in self.named_parameters())
        if all_param > 0:
            print(
                f"可训练 LoRA 参数: {trainable_params} || 总参数: {all_param} || 可训练比例%: {100 * trainable_params / all_param:.4f}"
            )
        else:
            print("模型中未找到参数。")
    
    def save_module(self, save_directory: str="direc_adapters"):
        """
        仅保存 Direc_Linear 模块的状态字典。
        """
        os.makedirs(save_directory, exist_ok=True)
        adapter_state_dicts = {}

        for name, module in self.named_modules():
            # 精确地识别 Direc_Linear 模块
            if isinstance(module, Direc_Linear):
                # 获取状态字典，确保在 CPU 上并且分离计算图
                # 注意：Direc_Linear 继承自 LoraLayer，其 state_dict 可能包含 base_layer
                # 如果只想保存 adapter 部分，需要过滤 state_dict
                # PEFT 通常使用 get_adapter_state_dict 方法，但我们是自定义的
                # 暂时保存完整的 state_dict，如果需要优化，则需过滤
                state_dict = module.state_dict()
                adapter_state_dicts[name] = {k: v.cpu().detach().clone() for k, v in state_dict.items()}
                logging.info(f"准备保存模块 {name} 的状态")

        # 将收集到的状态字典保存到一个文件中
        save_path = os.path.join(save_directory, "direc_adapter_states.pt")
        try:
            torch.save(adapter_state_dicts, save_path)
            logging.info(f"成功将适配器状态保存到 {save_path}")
        except Exception as e:
            logging.error(f"保存适配器状态失败: {e}")
            raise

    def load_module(self, load_directory: str="direc_adapters"):
        """
        从保存的文件加载 Direc_Linear 模块的状态字典，并确保 step 缓冲区为 long 类型。
        """
        load_path = os.path.join(load_directory, "direc_adapter_states.pt")

        if not os.path.exists(load_path):
            logging.error(f"错误: 适配器状态文件未找到于 {load_path}")
            raise FileNotFoundError(f"适配器状态文件未找到于 {load_path}")

        try:
            # 加载状态字典到 CPU
            adapter_state_dicts = torch.load(load_path, map_location='cpu')
            logging.info(f"从 {load_path} 加载了适配器状态")
        except Exception as e:
            logging.error(f"加载适配器状态文件失败: {e}")
            raise

        loaded_modules = set()
        all_direc_modules = {name for name, module in self.named_modules() if isinstance(module, Direc_Linear)}

        for name, module in self.named_modules():
            if isinstance(module, Direc_Linear):
                if name in adapter_state_dicts:
                    state_dict = adapter_state_dicts[name]

                    # --- 新增: 检查并强制转换 'step' 缓冲区类型 ---
                    buffer_key = 'step' # Direc_Linear 中注册的 buffer 名称
                    if buffer_key in state_dict and isinstance(state_dict[buffer_key], torch.Tensor):
                        if state_dict[buffer_key].dtype != torch.long:
                            original_dtype = state_dict[buffer_key].dtype
                            logging.warning(f"模块 {name} 的已加载 '{buffer_key}' 缓冲区类型为 {original_dtype}，将强制转换为 torch.long。")
                            # 强制转换为 long 类型
                            state_dict[buffer_key] = state_dict[buffer_key].long()
                        # 可选：如果需要确保它是一个标量（0维张量）
                        # if state_dict[buffer_key].ndim != 0:
                        #    logging.warning(f"模块 {name} 的 '{buffer_key}' 缓冲区不是标量，将尝试取第一个元素。")
                        #    state_dict[buffer_key] = state_dict[buffer_key].flatten()[0].long()

                    # ----------------------------------------------

                    # 获取当前模块的设备 (与之前代码相同)
                    try:
                       target_device = next(module.parameters()).device
                    except StopIteration:
                       try:
                           target_device = next(module.buffers()).device
                       except StopIteration:
                           target_device = torch.device('cpu')
                           logging.warning(f"模块 {name} 没有参数或缓冲区，状态字典将加载到 CPU")

                    # 将加载的状态字典中的张量移动到目标设备
                    state_dict = {k: v.to(target_device) for k, v in state_dict.items()}

                    try:
                        # 加载状态字典 (strict=False 推荐)
                        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
                        if missing_keys:
                             # 过滤掉可能因为不保存 base_layer 而出现的缺失键警告 (如果适用)
                             missing_adapter_keys = [k for k in missing_keys if any(p in k for p in Direc_Linear.adapter_layer_names + ('step',))] # 检查是否为 Direc_Linear 的关键部分
                             if missing_adapter_keys:
                                logging.warning(f"加载模块 {name} 时缺少关键适配器键: {missing_adapter_keys}")
                        if unexpected_keys:
                             # 过滤掉可能因为加载完整 state_dict 到只含适配器的模型而出现的意外键 (如果适用)
                             maybe_base_keys = [k for k in unexpected_keys if not any(p in k for p in Direc_Linear.adapter_layer_names + ('step',))]
                             if len(maybe_base_keys) < len(unexpected_keys): # 如果除了 base keys 外还有其他意外键
                                 other_unexpected = list(set(unexpected_keys) - set(maybe_base_keys))
                                 if other_unexpected:
                                     logging.warning(f"加载模块 {name} 时出现意外键: {other_unexpected}")

                        logging.info(f"成功加载模块 {name} 的状态字典")
                        loaded_modules.add(name)
                    except Exception as e:
                        logging.error(f"加载模块 {name} 的状态字典时出错: {e}")
                        # 可以选择继续或抛出异常
                else:
                    logging.warning(f"警告: 在加载文件中未找到模块 {name} 的保存状态。")

        # 检查是否有 Direc_Linear 模块未被加载 (与之前代码相同)
        unloaded_modules = all_direc_modules - loaded_modules
        if unloaded_modules:
            logging.warning(f"警告: 以下 Direc_Linear 模块存在但未从加载文件中找到其状态: {unloaded_modules}")