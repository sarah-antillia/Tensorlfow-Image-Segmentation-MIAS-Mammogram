# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import glob
from PIL import Image, ImageDraw
import shutil
import traceback

class MalignantImageMaskExtractor:

  def __init__(self):
    pass

  def extract(self,
    mammogram_master = "./MIASDBv1.21/",
    output_dir      = "./Mammogram/",
    images_dir      = "./Mammogram/images/",
    masks_dir       = "./Mammogram/masks/"):

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    os.makedirs(images_dir)
    os.makedirs(masks_dir)

    if os.path.exists(masks_dir):
      shutil.rmtree(masks_dir)
    os.makedirs(masks_dir)

    malignant_list = self.parse("./info.txt")
    #print(malignant_list)
    l = len(malignant_list)
    index = 1000
    for items in malignant_list:
      index += 1
      print(items)

      [filename, v1, v2, v3, x, y, r] =  items
      file_path = os.path.join(mammogram_master, filename+ ".pgm")
      image = Image.open(file_path)
      image = image.convert("RGB")
      w, h = image.size

      # Create an empty black mask of the same size of the image.
      mask = Image.new("RGB", (w, h)) 
      draw = ImageDraw.Draw(mask)
      x = int(x)
      y = int(y)
      r = int(r)
      x1 = (x - r) 
      y1 = (y - r) 
      x2 = (x + r) 
      y2 = (y + r)

      # Draw a red circle around the center (x, y) with the radius r.  
      draw.ellipse([(x1, y1), (x2, y2)], fill="red", outline="red", width=2)

      image_filename = os.path.join(images_dir,  filename + ".jpg")
      image.save(image_filename)

      mask_filename = os.path.join(masks_dir,  filename + ".jpg")
      mask.save(mask_filename)

    print("len {}".format(l))

  # Parse the info_txt file and get a list of malignant (marked by 'M') items. 
  def parse(self, info_txt):
    malignant_list = []
    with open(info_txt, "r") as f:
      lines = f.readlines()
    
      for line in lines:
        line = line.strip()
        if line.find("**") > 0:
            continue
        if line.find(" M ") >0:
          items = line.split()
          text = " ".join(items)
          items = text.split()
          print(items)
          malignant_list.append(items)
    return malignant_list


# Extract malignant images and create their corresponding masks
# They are relatively large images, probably over 4K pixels in width.
# Please remove the following jpg files from the generated Mamogram/images and masks folder.
# mdb072rm.jpg
# mdb110rl.jpg
# mdb124rm.jpg
# mdb130rl.jpg
# mdb267ll.jpg
# mdb270rm.jpg
# mdb271ll.jpg
# mdb274rx.jpg
#   
if __name__ == "__main__":
  try:
    extractor = MalignantImageMaskExtractor()
    
    extractor.extract(
      mammogram_master = "./MIASDBv1.21/",
      output_dir       = "./Mammogram/",
      images_dir       = "./Mammogram/images/",
      masks_dir        = "./Mammogram/masks/")

  except:
    traceback.print_exc()



