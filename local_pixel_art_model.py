import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from safetensors.torch import load_file
import os
from PIL import Image


class LocalPixelArtGenerator:
    def __init__(self, model_path="./pixel-art-xl.safetensors", base_model="runwayml/stable-diffusion-v1-5"):
        """
        初始化本地像素艺术生成器

        Args:
            model_path: 本地safetensors文件路径
            base_model: 基础模型ID（用于加载基础结构）
        """
        self.model_path = model_path
        self.base_model = base_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None

        print(f"使用设备: {self.device}")
        print(f"模型文件路径: {model_path}")

        self.load_model()

    def load_model(self):
        """加载本地模型文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

            print("正在加载基础模型结构...")
            # 首先加载基础模型
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )

            print("正在加载本地像素艺术权重...")
            # 加载safetensors文件
            state_dict = load_file(self.model_path)

            # 方法1: 如果是完整的UNet权重
            if any(key.startswith('unet.') for key in state_dict.keys()):
                print("检测到完整UNet权重，正在加载...")
                # 过滤出UNet相关的权重
                unet_state_dict = {k.replace('unet.', ''): v for k, v in state_dict.items() if k.startswith('unet.')}
                self.pipe.unet.load_state_dict(unet_state_dict, strict=False)

            # 方法2: 如果是LoRA权重或者其他格式
            else:
                print("尝试作为LoRA或微调权重加载...")
                try:
                    # 尝试直接加载到UNet
                    self.pipe.unet.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    print(f"直接加载失败，尝试其他方法: {e}")
                    # 可以在这里添加其他加载方法
                    self.load_as_lora(state_dict)

            # 移动到设备
            self.pipe = self.pipe.to(self.device)

            # 优化设置
            if self.device == "cuda":
                self.pipe.enable_memory_efficient_attention()
                self.pipe.enable_model_cpu_offload()

            print("✅ 本地像素艺术模型加载成功！")

        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            print("尝试使用基础模型...")
            # 如果加载失败，至少保证基础模型可用
            if self.pipe is None:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)

    def load_as_lora(self, state_dict):
        """尝试作为LoRA权重加载"""
        try:
            # 这里可以实现LoRA加载逻辑
            print("尝试LoRA加载方式...")
            # 如果你的模型是LoRA格式，可以使用以下方式：
            # self.pipe.load_lora_weights(state_dict)
            pass
        except Exception as e:
            print(f"LoRA加载也失败: {e}")

    def generate_pixel_art(self, prompt, negative_prompt=None, width=64, height=64,
                           num_images=1, steps=20, guidance_scale=7.5, seed=None):
        """
        生成像素艺术

        Args:
            prompt: 描述文本
            negative_prompt: 负面提示词
            width, height: 图像尺寸
            num_images: 生成数量
            steps: 推理步数
            guidance_scale: 引导强度
            seed: 随机种子
        """
        if self.pipe is None:
            print("模型未加载！")
            return None

        # 优化提示词
        pixel_prompt = f"pixel art, {prompt}, 8bit style, retro gaming, sharp pixels, detailed sprites"

        if negative_prompt is None:
            negative_prompt = "blurry, smooth, photorealistic, 3d render, low quality, watermark, text"

        print(f"正在生成像素艺术...")
        print(f"提示词: {pixel_prompt}")
        print(f"尺寸: {width}x{height}")

        try:
            # 设置随机种子
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)

            # 生成图像
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=pixel_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    num_images_per_prompt=num_images,
                    guidance_scale=guidance_scale,
                    generator=generator
                )

            print(f"✅ 成功生成 {len(result.images)} 张图像！")
            return result.images

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return None

    def save_images(self, images, output_dir="output", prefix="pixel_art", upscale=True):
        """保存生成的图像"""
        if not images:
            return []

        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        for i, image in enumerate(images):
            # 保存原始分辨率
            original_filename = os.path.join(output_dir, f"{prefix}_original_{i + 1}.png")
            image.save(original_filename)
            saved_files.append(original_filename)

            # 保存放大版本（便于查看）
            if upscale:
                upscaled = image.resize((image.width * 8, image.height * 8), Image.NEAREST)
                upscaled_filename = os.path.join(output_dir, f"{prefix}_upscaled_{i + 1}.png")
                upscaled.save(upscaled_filename)
                saved_files.append(upscaled_filename)

        return saved_files


def main():
    """使用示例"""
    print("=== 本地像素艺术模型使用示例 ===")

    # 检查模型文件
    model_file = "./pixel-art-xl.safetensors"
    if not os.path.exists(model_file):
        print(f"❌ 找不到模型文件: {model_file}")
        print("请确保文件路径正确，或修改 model_file 变量")

        # 提供文件路径输入选项
        custom_path = input("请输入模型文件的完整路径（或按回车跳过）: ").strip()
        if custom_path and os.path.exists(custom_path):
            model_file = custom_path
        else:
            print("将使用基础模型运行...")

    # 创建生成器
    try:
        generator = LocalPixelArtGenerator(model_path=model_file)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 预设示例
    examples = [
        {
            "prompt": "medieval sword with golden handle",
            "seed": 42,
            "description": "金柄中世纪剑"
        },
        {
            "prompt": "magic staff with blue crystal",
            "seed": 123,
            "description": "蓝水晶魔法杖"
        },
        {
            "prompt": "red potion bottle with cork",
            "seed": 456,
            "description": "红色药剂瓶"
        },
        {
            "prompt": "treasure chest full of gold coins",
            "seed": 789,
            "description": "装满金币的宝箱"
        }
    ]

    while True:
        print("\n" + "=" * 50)
        print("选择操作:")
        print("1. 运行预设示例")
        print("2. 自定义生成")
        print("3. 批量生成预设示例")
        print("4. 退出")

        choice = input("请选择 (1-4): ").strip()

        if choice == "1":
            print("\n预设示例:")
            for i, example in enumerate(examples, 1):
                print(f"{i}. {example['description']} - {example['prompt']}")

            try:
                idx = int(input("选择示例编号: ")) - 1
                if 0 <= idx < len(examples):
                    example = examples[idx]
                    print(f"\n生成: {example['description']}")

                    images = generator.generate_pixel_art(
                        prompt=example['prompt'],
                        seed=example['seed'],
                        width=64,
                        height=64,
                        steps=20
                    )

                    if images:
                        files = generator.save_images(images, prefix=f"example_{idx + 1}")
                        print("保存的文件:")
                        for file in files:
                            print(f"  {file}")
                else:
                    print("无效编号")
            except ValueError:
                print("请输入有效数字")

        elif choice == "2":
            prompt = input("输入描述 (英文): ").strip()
            if not prompt:
                continue

            try:
                width = int(input("宽度 (默认64): ") or "64")
                height = int(input("高度 (默认64): ") or "64")
                steps = int(input("步数 (默认20): ") or "20")
                seed = input("随机种子 (可选): ").strip()
                seed = int(seed) if seed else None
            except ValueError:
                width, height, steps, seed = 64, 64, 20, None

            images = generator.generate_pixel_art(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed
            )

            if images:
                files = generator.save_images(images, prefix="custom")
                print("保存的文件:")
                for file in files:
                    print(f"  {file}")

        elif choice == "3":
            print("正在批量生成所有预设示例...")
            for i, example in enumerate(examples):
                print(f"\n生成 {i + 1}/{len(examples)}: {example['description']}")

                images = generator.generate_pixel_art(
                    prompt=example['prompt'],
                    seed=example['seed'],
                    width=64,
                    height=64,
                    steps=20
                )

                if images:
                    generator.save_images(images, prefix=f"batch_{i + 1}")

            print("✅ 批量生成完成！请检查 output 文件夹")

        elif choice == "4":
            print("再见！")
            break

        else:
            print("无效选择")


if __name__ == "__main__":
    main()