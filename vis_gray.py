import os
import cv2
import sys
from tqdm import tqdm

def main():
    src_dir = sys.argv[1]
    save_dir = sys.argv[2]
    
    for file in tqdm(os.listdir(src_dir)):
        file_dir = os.path.join(src_dir, file)
        img = cv2.imread(file_dir)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(save_dir,("gray_{}.png").format(file)),img_gray)
        cv2.imwrite(os.path.join(save_dir,("color_{}.png").format(file)),gray_color)

if __name__ == '__main__':
    main()