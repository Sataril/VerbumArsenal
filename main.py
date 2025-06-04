from weapon_generator.model import WeaponGenerator
from weapon_generator.attributes import WeaponAttributes

def generate_weapon(description, resolution=(64,64)):
    # 文本预处理
    text_tensor = preprocess_text(description)
    
    # 生成图像
    model = WeaponGenerator()
    image_tensor = model(text_tensor)
    save_image(image_tensor, "weapon.png")
    
    # 生成属性
    attributes = WeaponAttributes(description)
    attributes.to_json("weapon_stats.json")

# ... existing helper functions ...