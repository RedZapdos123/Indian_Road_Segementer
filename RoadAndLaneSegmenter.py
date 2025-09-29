#This program combines YOLOPv2 for road and lane segmentation with YOLO11 for traffic object detection.
#A custom YOLO11 model fine tuned on the Indian driving dataset: Detection, is used by default, 
#but YOLOPv2 internal traffic detection can be enabled with --use-yolopv2-traffic.

import argparse
import time
from pathlib import Path
import cv2
import torch
import warnings
import os
from tqdm import tqdm
import random

#Suppress warnings and enable optimizations.
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  #Enable async CUDA operations for speed
torch.backends.cudnn.benchmark = True  #Optimize for consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  #Enable TF32 for faster computation

#mport YOLOPv2 modules.
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages

#Try to import YOLO11 for traffic object detection.
try:
    from ultralytics import YOLO
    YOLO11_AVAILABLE = True
except ImportError:
    YOLO11_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")
    print("Falling back to YOLOPv2 internal traffic detection.")

#YOLOPv2 class names (standard COCO classes for traffic objects).
YOLOPV2_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

#Bright color palette for class labels (BGR format for OpenCV).
BRIGHT_COLORS = [
    (0, 255, 255),    #Cyan
    (255, 0, 255),    #Magenta  
    (255, 255, 0),    #Yellow
    (0, 255, 0),      #Lime Green
    (255, 0, 0),      #Red
    (0, 165, 255),    #Orange
    (255, 20, 147),   #Deep Pink
    (0, 255, 127),    #Spring Green
    (255, 105, 180),  #Hot Pink
    (64, 224, 208),   #Turquoise
    (255, 215, 0),    #Gold
    (138, 43, 226),   #Blue Violet
    (50, 205, 50),    #Lime Green
    (255, 69, 0),     #Red Orange
    (0, 191, 255),    #Deep Sky Blue
    (255, 140, 0),    #Dark Orange
    (148, 0, 211),    #Dark Violet
    (0, 250, 154),    #Medium Spring Green
    (255, 20, 147),   #Deep Pink
    (32, 178, 170),   #Light Sea Green
]

#This function returns a bright color for a specific class ID.
def get_class_color(class_id, total_classes=None):
    if total_classes and len(BRIGHT_COLORS) < total_classes:
        #Generate additional colors if needed.
        random.seed(class_id)
        return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    return BRIGHT_COLORS[class_id % len(BRIGHT_COLORS)]


#This function creates the argument parser.
def make_parser():
    parser = argparse.ArgumentParser(description='Road and Lane Segmentation with Advanced Traffic Detection')
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='YOLOPv2 model path for segmentation and lanes')
    parser.add_argument('--traffic-weights', type=str, default='data/weights/IDDDYOLO11m.pt', help='YOLO11 model path for traffic objects (used by default)')
    parser.add_argument('--use-yolopv2-traffic', action='store_true', help='Use YOLOPv2 internal traffic detection instead of YOLO11 (not recommended due to accuracy issues)')
    parser.add_argument('--source', type=str, default='data/example.jpg', help='source')  #file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--max-frames', type=int, default=None, help='maximum frames to process for demo')
    return parser


#This function loads the YOLO11 model.
def yolo11_detect_traffic(model, img_bgr, conf_thres=0.3, iou_thres=0.45, device='cuda'):
    if model is None:
        return torch.empty((0, 6))
    
    try:
        #YOLO11 expects RGB image.
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        #Run YOLO11 inference.
        results = model.predict(
            img_rgb, 
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
            device=device
        )
        
        #Extract detections.
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            #Convert to tensor format [x1, y1, x2, y2, conf, class]
            detections = torch.cat([
                boxes.xyxy,  #x1, y1, x2, y2
                boxes.conf.unsqueeze(1),  #confidence
                boxes.cls.unsqueeze(1)   #class
            ], dim=1)
            return detections
        else:
            return torch.empty((0, 6))
            
    except Exception as e:
        print(f"YOLO11 detection error: {e}")
        return torch.empty((0, 6))

#This function loads the YOLOPv2 model.
def detect():
    source, weights,  save_txt, imgsz = opt.source, opt.weights,  opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  #save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  #increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  #make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    #Load YOLOPv2 model for segmentation and lane detection.
    stride = 32
    yolopv2_model = torch.jit.load(weights[0] if isinstance(weights, list) else weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'  #half precision only supported on CUDA.
    yolopv2_model = yolopv2_model.to(device)

    if half:
        yolopv2_model.half()
    yolopv2_model.eval()

    #Load YOLO11 model for traffic object detection by default.
    yolo11_model = None
    use_yolopv2_traffic = opt.use_yolopv2_traffic  #Only use YOLOPv2 if explicitly requested.
    use_yolo11_traffic = not use_yolopv2_traffic and YOLO11_AVAILABLE  #Use YOLO11 by default.
    yolo11_names = None
    
    if use_yolo11_traffic:
        if not Path(opt.traffic_weights).exists():
            print(f"Warning: YOLO11 traffic weights not found at {opt.traffic_weights}")
            print("Falling back to YOLOPv2 internal traffic detection.")
            use_yolo11_traffic = False
            use_yolopv2_traffic = True
        else:
            try:
                yolo11_model = YOLO(opt.traffic_weights)
                yolo11_model.to(device)
                yolo11_names = yolo11_model.names  #Get class names from YOLO11 model
                print(f"Loaded YOLO11 traffic model: {opt.traffic_weights}")
            except Exception as e:
                print(f"Failed to load YOLO11 model: {e}")
                print("Falling back to YOLOPv2 internal traffic detection.")
                use_yolo11_traffic = False
                use_yolopv2_traffic = True
    else:
        print("Using YOLOPv2 internal traffic detection (as requested)")
    
    if not YOLO11_AVAILABLE and not use_yolopv2_traffic:
        print("YOLO11 not available, using YOLOPv2 internal traffic detection.")
        use_yolopv2_traffic = True
        use_yolo11_traffic = False

    #GPU memory optimization.
    if device.type != 'cpu':
        torch.cuda.empty_cache()
        #Warm up YOLOPv2 model.
        dummy_input = torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(yolopv2_model.parameters()))
        for _ in range(3):  #Multiple warmup runs.
            with torch.no_grad():
                _ = yolopv2_model(dummy_input)
        torch.cuda.synchronize()
        
        #Warm up YOLO11 model if loaded.
        if use_yolo11_traffic and yolo11_model:
            dummy_img = torch.randint(0, 255, (imgsz, imgsz, 3), dtype=torch.uint8).cpu().numpy()
            yolo11_model.predict(dummy_img, verbose=False)  #Warm up YOLO11

    #Set Dataloader.
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    #Initialize video processing variables.
    frame_count = 0
    total_frames = getattr(dataset, 'nframes', None) if hasattr(dataset, 'video_flag') and any(dataset.video_flag) else len(dataset)
    max_frames = min(opt.max_frames, total_frames) if opt.max_frames and total_frames else opt.max_frames

    #Setup progress bar.
    progress_desc = "Processing"
    if dataset.mode == 'video':
        progress_desc = f"Processing {Path(source).name}"
    
    #Run inference.
    if device.type != 'cpu':
        yolopv2_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(yolopv2_model.parameters())))  #run once
    t0 = time.time()
    
    #Pre initialize video writer info to avoid printing during processing.
    vid_setup_done = False
    
    pbar = tqdm(total=max_frames if max_frames else total_frames, 
                desc=progress_desc, 
                unit="frames",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for path, img, im0s, vid_cap in dataset:
        if max_frames and frame_count >= max_frames:
            pbar.write(f"Reached maximum frames limit ({max_frames}). Stopping processing.")
            pbar.close()
            break
            
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  #uint8 to fp16/32
        img /= 255.0  #0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        #YOLOPv2 inference for segmentation and lane detection.
        t1 = time_synchronized()
        with torch.no_grad():  #Optimize inference.
            [pred,anchor_grid],seg,ll= yolopv2_model(img)
        t2 = time_synchronized()

        #Get segmentation masks from YOLOPv2.
        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        #Traffic object detection: use YOLO11 by default, YOLOPv2 if explicitly requested.
        if use_yolo11_traffic and yolo11_model:
            #Use YOLO11 for traffic object detection.
            yolo11_detections = yolo11_detect_traffic(
                yolo11_model, 
                im0s, 
                conf_thres=opt.conf_thres,
                iou_thres=opt.iou_thres,
                device=device
            )
            
            #Convert YOLO11 detections to YOLOPv2 format for consistent processing.
            if len(yolo11_detections) > 0:
                #Scale detections to match YOLOPv2 input size.
                yolo11_detections[:, :4] = scale_coords(
                    (im0s.shape[0], im0s.shape[1]), 
                    yolo11_detections[:, :4], 
                    img.shape[2:]
                ).round()
                #Wrap in list to match YOLOPv2 format.
                pred = [yolo11_detections]
            else:
                pred = [torch.empty((0, 6))]
                
            #Skip YOLOPv2 traffic detection processing.
            tw1 = tw2 = t3 = t4 = time_synchronized()
            
        else:
            tw1 = time_synchronized()
            pred = split_for_trace_model(pred,anchor_grid)
            tw2 = time_synchronized()

            #Apply NMS.
            t3 = time_synchronized()
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t4 = time_synchronized()
        
        frame_count += 1

        #Process detections.
        for i, det in enumerate(pred):
          
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name) 
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  #img.txt
            s += '%gx%g ' % img.shape[2:] 
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum() 
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  #add to string

                for *xyxy, conf, cls in reversed(det):
                    class_id = int(cls)
                    
                    #Get class name and color based on which model is being used.
                    if use_yolo11_traffic and yolo11_names:
                        class_name = yolo11_names.get(class_id, f'class{class_id}')
                        total_classes = len(yolo11_names)
                    else:
                        class_name = YOLOPV2_NAMES.get(class_id, f'class{class_id}')
                        total_classes = len(YOLOPV2_NAMES)
                    
                    #Create label with class name only (no confidence score).
                    label = f'{class_name}'
                    color = get_class_color(class_id, total_classes)
                    
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh) 
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img :
                        plot_one_box(xyxy, im0, color=color, label=label, line_thickness=3)

            #Apply segmentation visualization.
            show_seg_result(im0, (da_seg_mask,ll_seg_mask), is_demo=True)
            
            #Print time (inference) - only for images to reduce spam.
            if dataset.mode == 'image':
                print(f'{s}Done. ({t2 - t1:.3f}s)')

            #Save results (image with detections).
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:
                    if vid_path != save_path: 
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap: 
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w,h = im0.shape[1], im0.shape[0]

                            if not save_path.endswith('.mp4'):
                                save_path = str(Path(save_path).with_suffix('.mp4'))
                        else:  
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            if not save_path.endswith('.mp4'):
                                save_path += '.mp4'
                        
                        #Try different codecs for better compatibility
                        codecs_to_try = [
                            ('mp4v', 'MP4V'),  #Most compatible
                            ('XVID', 'XVID'),  #Good compatibility
                            ('MJPG', 'MJPG'),  #Fallback option
                        ]
                        
                        vid_writer = None
                        for codec_str, codec_name in codecs_to_try:
                            try:
                                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                                test_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                                if test_writer.isOpened():
                                    vid_writer = test_writer
                                    pbar.write(f"Using {codec_name} codec for video output")
                                    break
                                else:
                                    test_writer.release()
                            except Exception as e:
                                if 'test_writer' in locals():
                                    test_writer.release()
                                continue
                        
                        #Final fallback if all codecs fail.
                        if vid_writer is None or not vid_writer.isOpened():
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                            pbar.write("Using default mp4v codec")
                        
                        #Print video setup info only once, before progress bar starts.
                        if not vid_setup_done:
                            pbar.write(f"Processing video: {p.name} -> {save_path}")
                            pbar.write(f"Video properties: {w}x{h} @ {fps:.1f} FPS")
                            if max_frames:
                                pbar.write(f"Processing {max_frames} frames out of {total_frames} total frames")
                            vid_setup_done = True
                    vid_writer.write(im0)

        #Update progress bar with real time info.
        if dataset.mode == 'video':
            current_fps = 1 / (t2 - t1) if (t2 - t1) > 0 else 0
            pbar.set_postfix({
                'FPS': f'{current_fps:.1f}',
                'Frame': f'{frame_count}/{max_frames if max_frames else total_frames}'
            })
        pbar.update(1)

    #Close progress bar.
    pbar.close()
    
    #Clean up video writer.
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
        pbar.write(f"Video processing completed! Output saved to: {vid_path}")

    batch_size = 1
    inf_time.update(t2-t1, batch_size)
    nms_time.update(t4-t3, batch_size)
    waste_time.update(tw2-tw1, batch_size)
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')

#This is the main entry point for the script.
if __name__ == '__main__':
    opt =  make_parser().parse_args()
    print(opt)

    with torch.no_grad():
            detect()