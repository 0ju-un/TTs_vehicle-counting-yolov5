import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.utils.general import (check_requirements, print_args)
from track import (load_model, run)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5s-seg.pt', help='model.pt path(s)')
    parser.add_argument('--reid_weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking_method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--green_time', type=int,default=10, help='circular green light time')
    parser.add_argument('--arrow_time', type=int,default=10, help='green arrow light time')
    parser.add_argument('--red_time', type=int,default=10, help='red light time')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    device, model, imgsz = load_model(yolo_weights=opt.yolo_weights,imgsz=opt.imgsz,device=opt.device)
    print('model loaded')
    green_time = opt.green_time
    arrow_time = opt.arrow_time
    red_time = opt.red_time

    green_count = run(device=device, model=model, imgsz=imgsz,
        reid_weights=opt.reid_weights,
        tracking_method=opt.tracking_method,
        classes=opt.classes,
        light_time=green_time)
    print(f'green_count : {green_count}')
