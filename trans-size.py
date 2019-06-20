
import os.path
import glob
from PIL import Image

filename = os.listdir("./trans-size/zh_zhe")
base_dir = "./trans-size/zh_zhe/*.jpg"
new_dir = "./trans-size/new"

for jpgfile in glob.glob(base_dir):
    img = Image.open(jpgfile)
    new_img = img.resize((32, 40), Image.BILINEAR)
    new_img.save(os.path.join(new_dir, os.path.basename(jpgfile)))
