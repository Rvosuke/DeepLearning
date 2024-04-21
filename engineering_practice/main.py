from PIL import Image, ImageSequence


def remove_watermark(input_gif, output_gif):
    with Image.open(input_gif) as im:
        frames = []
        for frame in ImageSequence.Iterator(im):
            # 将每一帧转换为RGBA模式
            frame = frame.convert("RGBA")

            # 获取帧的像素数据
            data = frame.getdata()

            # 创建一个新的像素列表
            new_data = []
            for item in data:
                # 如果像素接近白色,则将其设置为完全透明
                if item[0] > 220 and item[1] > 220 and item[2] > 220:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)

            # 更新帧的像素数据
            frame.putdata(new_data)

            # 将处理后的帧添加到列表中
            frames.append(frame)

        # 保存去除水印后的GIF图像
        frames[0].save(output_gif, save_all=True, append_images=frames[1:], loop=0)


# 使用示例
remove_watermark("input.gif", "output.gif")