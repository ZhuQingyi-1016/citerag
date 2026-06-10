import argparse
import os

import cv2


def center_crop_square(img):
    h, w = img.shape[:2]
    m = min(h, w)
    y1 = (h - m) // 2
    x1 = (w - m) // 2
    return img[y1:y1+m, x1:x1+m]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--rotate", type=int, default=90)  # 90/180/270
    args = parser.parse_args()

    img = cv2.imread(args.in_path)
    if img is None:
        raise ValueError(f"Cannot read image: {args.in_path}")

    img = center_crop_square(img)

    if args.rotate == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif args.rotate == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif args.rotate == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    cv2.imwrite(args.out_path, img)
    print(f"Saved to {args.out_path}")

if __name__ == "__main__":
    main()