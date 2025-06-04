import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, EulerAncestralDiscreteScheduler, \
    DPMSolverMultistepScheduler
from safetensors.torch import load_file
import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


class EnhancedPixelArtGenerator:
    def __init__(self, model_path="./pixel_art_model.safetensors", base_model_path=None):
        """
        增强版像素艺术生成器
        """
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None

        print(f"使用设备: {self.device}")
        print(f"像素艺术模型文件路径: {model_path}")

        self.load_model()
        self.setup_optimizations()

    def load_model(self):
        """加载模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

            print("正在加载基础模型...")

            # 使用更好的基础模型
            base_models = [
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2-1-base",
                "CompVis/stable-diffusion-v1-4"
            ]

            for base_model in base_models:
                try:
                    self.pipe = StableDiffusionPipeline.from_pretrained(
                        base_model,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True
                    )
                    print(f"✅ 成功加载基础模型: {base_model}")
                    break
                except Exception as e:
                    print(f"尝试 {base_model} 失败: {e}")
                    continue

            if self.pipe is None:
                raise Exception("无法加载任何基础模型")

            # 加载像素艺术权重
            print("正在加载像素艺术权重...")
            state_dict = load_file(self.model_path)
            self.load_weights_advanced(state_dict)

            # 移动到设备
            self.pipe = self.pipe.to(self.device)

            print("✅ 模型加载完成！")

        except Exception as e:
            print(f"❌ 加载失败: {e}")
            self.pipe = None

    def load_weights_advanced(self, state_dict):
        """高级权重加载方法"""
        try:
            print(f"权重文件包含 {len(state_dict)} 个参数")

            # 分析权重结构
            key_patterns = {}
            for key in state_dict.keys():
                pattern = key.split('.')[0] if '.' in key else key
                key_patterns[pattern] = key_patterns.get(pattern, 0) + 1

            print("权重结构分析:")
            for pattern, count in key_patterns.items():
                print(f"  {pattern}: {count} 个参数")

            # 尝试多种加载方式
            success = False

            # 方法1: 直接加载到UNet
            try:
                missing, unexpected = self.pipe.unet.load_state_dict(state_dict, strict=False)
                if len(missing) < len(state_dict) * 0.8:  # 如果缺失的键不太多
                    print(f"✅ UNet加载成功 (缺失: {len(missing)}, 意外: {len(unexpected)})")
                    success = True
            except Exception as e:
                print(f"UNet直接加载失败: {e}")

            # 方法2: 清理键名后加载
            if not success:
                try:
                    cleaned_dict = {}
                    for k, v in state_dict.items():
                        # 移除常见前缀
                        new_key = k
                        prefixes_to_remove = ['model.', 'unet.', 'diffusion_model.']
                        for prefix in prefixes_to_remove:
                            if new_key.startswith(prefix):
                                new_key = new_key[len(prefix):]
                        cleaned_dict[new_key] = v

                    missing, unexpected = self.pipe.unet.load_state_dict(cleaned_dict, strict=False)
                    print(f"✅ 清理后加载成功 (缺失: {len(missing)}, 意外: {len(unexpected)})")
                    success = True
                except Exception as e:
                    print(f"清理后加载失败: {e}")

            if not success:
                print("⚠️ 权重加载可能不完整，但会尝试继续运行")

        except Exception as e:
            print(f"权重加载过程出错: {e}")

    def setup_optimizations(self):
        """设置各种优化"""
        if self.pipe is None:
            return

        try:
            # 使用更好的调度器
            print("设置高质量调度器...")
            # DPM++ 调度器通常效果更好
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            # CUDA优化
            if self.device == "cuda":
                try:
                    self.pipe.enable_memory_efficient_attention()
                    print("✅ 启用内存高效注意力")
                except:
                    pass

                try:
                    # 使用xformers如果可用
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("✅ 启用xformers优化")
                except:
                    pass

                # 编译模型以提高性能（PyTorch 2.0+）
                try:
                    if hasattr(torch, 'compile'):
                        self.pipe.unet = torch.compile(self.pipe.unet)
                        print("✅ 启用模型编译优化")
                except:
                    pass

        except Exception as e:
            print(f"优化设置失败: {e}")

    def get_optimized_prompts(self, prompt):
        """获取优化的提示词"""
        # 高质量像素艺术提示词模板
        quality_terms = [
            "masterpiece", "best quality", "high quality", "detailed",
            "sharp", "crisp", "clean", "professional"
        ]

        style_terms = [
            "pixel art", "8-bit", "16-bit", "retro gaming style",
            "sprite art", "game asset", "pixelated", "digital art"
        ]

        technical_terms = [
            "sharp pixels", "no blur", "distinct pixels", "grid-based",
            "limited color palette", "dithering", "clean edges"
        ]

        # 构建增强提示词
        enhanced_prompt = f"{', '.join(quality_terms[:3])}, {', '.join(style_terms[:4])}, {prompt}, {', '.join(technical_terms[:3])}"

        # 负面提示词
        negative_prompt = (
            "blurry, smooth, soft, photorealistic, 3d render, realistic, "
            "anti-aliasing, gradient, noise, jpeg artifacts, low quality, "
            "worst quality, normal quality, lowres, watermark, signature, "
            "text, logo, bad anatomy, distorted, ugly, deformed"
        )

        return enhanced_prompt, negative_prompt

    def generate_pixel_art(self, prompt, negative_prompt=None, width=512, height=512,
                           num_images=1, steps=30, guidance_scale=8.5, seed=None,
                           enhance_mode="balanced"):
        """
        生成像素艺术

        Args:
            enhance_mode: "fast", "balanced", "quality"
        """
        if self.pipe is None:
            print("❌ 模型未加载！")
            return None

        # 获取优化的提示词
        enhanced_prompt, default_negative = self.get_optimized_prompts(prompt)
        if negative_prompt is None:
            negative_prompt = default_negative

        # 根据增强模式调整参数
        if enhance_mode == "quality":
            steps = max(steps, 40)
            guidance_scale = max(guidance_scale, 9.0)
            width = max(width, 768)
            height = max(height, 768)
        elif enhance_mode == "fast":
            steps = min(steps, 20)
            guidance_scale = min(guidance_scale, 7.0)

        print(f"正在生成像素艺术 ({enhance_mode} 模式)...")
        print(f"增强提示词: {enhanced_prompt[:100]}...")
        print(f"参数: {width}x{height}, {steps}步, 引导:{guidance_scale}")

        try:
            # 设置随机种子
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)

            # 生成图像
            with torch.autocast(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32):
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    num_images_per_prompt=num_images,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    eta=0.0  # 确定性生成
                )

            print(f"✅ 成功生成 {len(result.images)} 张图像！")

            # 后处理增强
            enhanced_images = []
            for img in result.images:
                enhanced_img = self.post_process_image(img, enhance_mode)
                enhanced_images.append(enhanced_img)

            return enhanced_images

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def post_process_image(self, image, enhance_mode="balanced"):
        """后处理图像以增强像素艺术效果"""
        try:
            # 转换为RGB确保兼容性
            if image.mode != 'RGB':
                image = image.convert('RGB')

            if enhance_mode == "quality":
                # 高质量后处理
                # 1. 轻微锐化
                image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

                # 2. 增强对比度
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)

                # 3. 增强饱和度
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.1)

            elif enhance_mode == "balanced":
                # 平衡处理
                # 轻微锐化
                image = image.filter(ImageFilter.UnsharpMask(radius=0.5, percent=120, threshold=2))

                # 轻微增强对比度
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)

            # 可选：像素化处理（使原本不够像素化的图像更像素化）
            if enhance_mode in ["quality", "balanced"]:
                # 缩小再放大来增强像素效果
                small_size = (image.width // 4, image.height // 4)
                small_img = image.resize(small_size, Image.NEAREST)
                image = small_img.resize((image.width, image.height), Image.NEAREST)

            return image

        except Exception as e:
            print(f"后处理失败: {e}")
            return image

    def save_images(self, images, output_dir="output", prefix="pixel_art",
                    upscale_factors=[1, 2, 4, 8], save_formats=['PNG']):
        """保存生成的图像，支持多种放大倍数和格式"""
        if not images:
            return []

        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        for i, image in enumerate(images):
            base_filename = f"{prefix}_{i + 1}"

            for factor in upscale_factors:
                if factor == 1:
                    # 原始尺寸
                    for fmt in save_formats:
                        filename = os.path.join(output_dir, f"{base_filename}_original.{fmt.lower()}")
                        if fmt.upper() == 'PNG':
                            image.save(filename, format='PNG', optimize=True)
                        elif fmt.upper() == 'WEBP':
                            image.save(filename, format='WEBP', quality=95, lossless=True)
                        saved_files.append(filename)
                else:
                    # 放大版本
                    upscaled = image.resize(
                        (image.width * factor, image.height * factor),
                        Image.NEAREST  # 保持像素艺术的锐利边缘
                    )
                    for fmt in save_formats:
                        filename = os.path.join(output_dir, f"{base_filename}_{factor}x.{fmt.lower()}")
                        if fmt.upper() == 'PNG':
                            upscaled.save(filename, format='PNG', optimize=True)
                        elif fmt.upper() == 'WEBP':
                            upscaled.save(filename, format='WEBP', quality=95, lossless=True)
                        saved_files.append(filename)

        return saved_files

    def benchmark_settings(self, prompt="simple sword", test_configs=None):
        """测试不同设置的效果"""
        if test_configs is None:
            test_configs = [
                {"steps": 20, "guidance": 7.0, "size": (256, 256), "mode": "fast"},
                {"steps": 30, "guidance": 8.5, "size": (512, 512), "mode": "balanced"},
                {"steps": 40, "guidance": 9.0, "size": (768, 768), "mode": "quality"},
            ]

        print("=== 设置基准测试 ===")
        results = []

        for i, config in enumerate(test_configs):
            print(f"\n测试配置 {i + 1}: {config}")

            import time
            start_time = time.time()

            images = self.generate_pixel_art(
                prompt=prompt,
                width=config["size"][0],
                height=config["size"][1],
                steps=config["steps"],
                guidance_scale=config["guidance"],
                enhance_mode=config["mode"],
                seed=42  # 固定种子便于比较
            )

            end_time = time.time()
            duration = end_time - start_time

            if images:
                # 保存测试结果
                files = self.save_images(
                    images,
                    output_dir="benchmark_output",
                    prefix=f"test_{i + 1}_{config['mode']}",
                    upscale_factors=[1, 4]
                )
                results.append({
                    "config": config,
                    "duration": duration,
                    "files": files
                })
                print(f"✅ 完成，耗时: {duration:.2f}秒")
            else:
                print("❌ 生成失败")

        return results


def main():
    """主程序"""
    print("=== 增强版像素艺术生成器 ===")

    # 查找模型文件
    possible_names = [
        "./pixel_art_model.safetensors",
        "./pixel-art-model.safetensors",
        "./pixel-art-xl.safetensors",
        "./pixelart.safetensors"
    ]

    found_model = None
    for name in possible_names:
        if os.path.exists(name):
            found_model = name
            break

    if not found_model:
        print("❌ 找不到模型文件")
        custom_path = input("请输入模型文件路径: ").strip()
        if custom_path and os.path.exists(custom_path):
            found_model = custom_path
        else:
            print("无效路径，程序退出")
            return

    # 创建生成器
    try:
        generator = EnhancedPixelArtGenerator(model_path=found_model)
        if generator.pipe is None:
            print("❌ 模型加载失败")
            return
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 高质量预设示例
    premium_examples = [
        {
            "prompt": "legendary crystal sword with glowing blue blade and silver hilt",
            "seed": 42,
            "description": "传奇水晶剑",
            "mode": "quality"
        },
        {
            "prompt": "ancient magic staff topped with burning phoenix feather",
            "seed": 123,
            "description": "凤凰法杖",
            "mode": "quality"
        },
        {
            "prompt": "enchanted bow made of living wood with golden string",
            "seed": 456,
            "description": "魔法弓",
            "mode": "balanced"
        },
        {
            "prompt": "cursed black armor with red glowing runes",
            "seed": 789,
            "description": "诅咒铠甲",
            "mode": "quality"
        }
    ]

    while True:
        print("\n" + "=" * 60)
        print("选择操作:")
        print("1. 生成高质量示例")
        print("2. 自定义生成（基础）")
        print("3. 自定义生成（高级）")
        print("4. 批量生成示例")
        print("5. 设置基准测试")
        print("6. 退出")

        choice = input("请选择 (1-6): ").strip()

        if choice == "1":
            print("\n高质量预设示例:")
            for i, example in enumerate(premium_examples, 1):
                print(f"{i}. {example['description']} ({example['mode']} 模式)")

            try:
                idx = int(input("选择示例编号: ")) - 1
                if 0 <= idx < len(premium_examples):
                    example = premium_examples[idx]
                    print(f"\n生成: {example['description']}")

                    images = generator.generate_pixel_art(
                        prompt=example['prompt'],
                        seed=example['seed'],
                        width=512,
                        height=512,
                        steps=35,
                        guidance_scale=8.5,
                        enhance_mode=example['mode']
                    )

                    if images:
                        files = generator.save_images(
                            images,
                            prefix=f"premium_{idx + 1}",
                            upscale_factors=[1, 2, 4, 8],
                            save_formats=['PNG', 'WEBP']
                        )
                        print("保存的文件:")
                        for file in files[:4]:  # 只显示前几个
                            print(f"  {file}")
                        if len(files) > 4:
                            print(f"  ... 还有 {len(files) - 4} 个文件")
                else:
                    print("无效编号")
            except ValueError:
                print("请输入有效数字")

        elif choice == "2":
            # 基础自定义生成
            prompt = input("输入描述 (英文): ").strip()
            if not prompt:
                continue

            images = generator.generate_pixel_art(
                prompt=prompt,
                width=512,
                height=512,
                steps=30,
                guidance_scale=8.5,
                enhance_mode="balanced"
            )

            if images:
                files = generator.save_images(images, prefix="custom_basic")
                print("保存的文件:")
                for file in files:
                    print(f"  {file}")

        elif choice == "3":
            # 高级自定义生成
            prompt = input("输入描述 (英文): ").strip()
            if not prompt:
                continue

            try:
                print("\n高级设置:")
                width = int(input("宽度 (推荐512): ") or "512")
                height = int(input("高度 (推荐512): ") or "512")
                steps = int(input("步数 (20-50, 推荐35): ") or "35")
                guidance = float(input("引导强度 (7.0-12.0, 推荐8.5): ") or "8.5")

                print("增强模式: 1=快速, 2=平衡, 3=高质量")
                mode_choice = input("选择模式 (默认2): ") or "2"
                modes = {"1": "fast", "2": "balanced", "3": "quality"}
                enhance_mode = modes.get(mode_choice, "balanced")

                seed = input("随机种子 (可选): ").strip()
                seed = int(seed) if seed else None

                num_images = int(input("生成数量 (1-4, 默认1): ") or "1")
                num_images = min(max(num_images, 1), 4)

            except ValueError:
                print("参数错误，使用默认设置")
                width, height, steps, guidance = 512, 512, 35, 8.5
                enhance_mode, seed, num_images = "balanced", None, 1

            images = generator.generate_pixel_art(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance,
                enhance_mode=enhance_mode,
                seed=seed,
                num_images=num_images
            )

            if images:
                files = generator.save_images(
                    images,
                    prefix="custom_advanced",
                    upscale_factors=[1, 2, 4, 8]
                )
                print("保存的文件:")
                for file in files:
                    print(f"  {file}")

        elif choice == "4":
            print("正在批量生成高质量示例...")
            for i, example in enumerate(premium_examples):
                print(f"\n生成 {i + 1}/{len(premium_examples)}: {example['description']}")

                images = generator.generate_pixel_art(
                    prompt=example['prompt'],
                    seed=example['seed'],
                    width=512,
                    height=512,
                    steps=35,
                    enhance_mode=example['mode']
                )

                if images:
                    generator.save_images(images, prefix=f"batch_premium_{i + 1}")

            print("✅ 批量生成完成！")

        elif choice == "5":
            print("开始设置基准测试...")
            results = generator.benchmark_settings()

            print("\n=== 基准测试结果 ===")
            for i, result in enumerate(results):
                config = result['config']
                print(f"\n配置 {i + 1} ({config['mode']} 模式):")
                print(f"  分辨率: {config['size']}")
                print(f"  步数: {config['steps']}")
                print(f"  耗时: {result['duration']:.2f}秒")
                print(f"  文件: {len(result['files'])} 个")

        elif choice == "6":
            print("再见！")
            break

        else:
            print("无效选择")


if __name__ == "__main__":
    main()