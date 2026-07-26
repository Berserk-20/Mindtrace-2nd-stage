import os
from PIL import Image, ImageDraw, ImageFont

images = [
    [("backend/dataset/train/happy/Training_10019449.jpg", "Happy (Tr)"),
     ("backend/dataset/train/angry/Training_10118481.jpg", "Angry (Tr)"),
     ("backend/dataset/train/surprise/Training_10013223.jpg", "Surprise (Tr)"),
     ("backend/dataset/train/neutral/Training_10002154.jpg", "Neutral (Tr)"),
     ("backend/dataset/train/sad/Training_10022789.jpg", "Sad (Tr)"),
     ("backend/dataset/train/fear/Training_10018621.jpg", "Fear (Tr)"),
     ("backend/dataset/train/disgust/Training_10371709.jpg", "Disgust (Tr)")],
    [("backend/dataset/test/happy/PrivateTest_10077120.jpg", "Happy (Te)"),
     ("backend/dataset/test/angry/PrivateTest_10131363.jpg", "Angry (Te)"),
     ("backend/dataset/test/surprise/PrivateTest_10072988.jpg", "Surprise (Te)"),
     ("backend/dataset/test/neutral/PrivateTest_10086748.jpg", "Neutral (Te)"),
     ("backend/dataset/test/sad/PrivateTest_10247676.jpg", "Sad (Te)"),
     ("backend/dataset/test/fear/PrivateTest_10153550.jpg", "Fear (Te)"),
     ("backend/dataset/test/disgust/PrivateTest_11895083.jpg", "Disgust (Te)")]
]

img_size = 100
padding = 10
text_height = 25
cols = 7
rows = 2

width = cols * img_size + (cols + 1) * padding
height = rows * (img_size + text_height) + (rows + 1) * padding

canvas = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("arial.ttf", 14)
except IOError:
    font = ImageFont.load_default()

for r, row in enumerate(images):
    for c, (img_path, label) in enumerate(row):
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).resize((img_size, img_size))
                x = padding + c * (img_size + padding)
                y = padding + r * (img_size + text_height + padding)
                canvas.paste(img, (x, y))
                
                # Center text
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_x = x + (img_size - text_w) // 2
                text_y = y + img_size + 5
                draw.text((text_x, text_y), label, fill="black", font=font)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        else:
            print(f"Missing: {img_path}")

canvas.save("dataset_grid.png")
print("Saved dataset_grid.png")
