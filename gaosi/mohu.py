from PIL import Image, ImageFilter
import os

# 图片所在的文件夹
input_folder = 'images'
# 输出的文件夹
output_folder = 'xml'

# 创建输出文件夹如果它不存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有图片
for image_file in os.listdir(input_folder):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        # 打开图片
        img = Image.open(os.path.join(input_folder, image_file))
        # 应用高斯模糊效果
        blurred_img = img.filter(ImageFilter.GaussianBlur(25))
        # 保存处理后的图片到输出文件夹
        blurred_img.save(os.path.join(output_folder, image_file))

print("所有图片已处理完毕。")