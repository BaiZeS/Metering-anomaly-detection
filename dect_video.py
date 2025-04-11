# 代码框架：基于图像的计量异常监测系统
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


# 全局变量
cap = None
roi = None
des_base = None
mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
last_frame = None


# 阈值参数
FG_RATIO_THRESH = 0.01
MATCH_RATIO_THRESH = 0.7
MOTION_THRESH = 0.05

# 在全局变量部分新增
roi_points = []
selecting_roi = False

# 新增鼠标回调函数
def select_roi(event, x, y, flags, param):
    """ROI手工标定"""
    global roi_points, selecting_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        selecting_roi = True
    elif event == cv2.EVENT_LBUTTONUP and selecting_roi:
        roi_points.append((x, y))
        selecting_roi = False


def preprocess(frame):
    """基础预处理：灰度化+直方图均衡化"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def enhanced_preprocess(img):
    """增强预处理：CLAHE对比度增强"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)


def dynamic_difference_detection(base_img, current_img, roi=None):
    """动态差异检测"""
    global last_frame
    
    # 获取ROI坐标（若未设置则使用全图）
    x, y, w, h = roi or (0, 0, current_img.shape[1], current_img.shape[0])
    
    # 提取ROI区域
    base_roi = base_img[y:y+h, x:x+w]
    current_roi = current_img[y:y+h, x:x+w]
    
    # 预处理流程（仅处理ROI区域）
    base_enhanced = enhanced_preprocess(base_roi)
    current_enhanced = enhanced_preprocess(current_roi)
    
    # 新增颜色差异检测（LAB颜色空间）
    base_lab = cv2.cvtColor(base_enhanced, cv2.COLOR_BGR2LAB)
    current_lab = cv2.cvtColor(current_enhanced, cv2.COLOR_BGR2LAB)
    a_diff = cv2.absdiff(base_lab[:,:,1], current_lab[:,:,1])
    b_diff = cv2.absdiff(base_lab[:,:,2], current_lab[:,:,2])
    color_diff = cv2.bitwise_or(a_diff, b_diff)
    _, color_mask = cv2.threshold(cv2.medianBlur(color_diff,5), 15, 255, cv2.THRESH_BINARY)
    
    # 静态帧差检测（ROI内部）
    base_gray = cv2.cvtColor(base_enhanced, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_enhanced, cv2.COLOR_BGR2GRAY)
    static_diff = cv2.absdiff(base_gray, current_gray)
    _, static_mask = cv2.threshold(cv2.GaussianBlur(static_diff, (3,3), 0), 30, 255, cv2.THRESH_BINARY)
    
    # 寻找最大连通区域
    contours, _ = cv2.findContours(static_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_contour = cnt
    
    # 创建主检测区域掩码
    main_mask = np.zeros_like(static_mask)
    if max_contour is not None:
        # 对最大轮廓进行形态学优化
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        cv2.drawContours(main_mask, [approx], -1, 255, -1)
        main_mask = cv2.morphologyEx(main_mask, cv2.MORPH_CLOSE, 
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
    
    # 动态检测（聚焦主区域）
    gray_roi = preprocess(current_enhanced)
    gray_roi = cv2.GaussianBlur(gray_roi, (3,3), 0)
    fg_mask = mog2.apply(gray_roi)
    fg_mask = cv2.bitwise_and(fg_mask, main_mask)  # 只保留主区域动态变化
    
    # 融合检测结果（新增颜色差异）
    combined_mask = cv2.bitwise_or(fg_mask, main_mask)
    combined_mask = cv2.bitwise_or(combined_mask, color_mask)  # 新增颜色差异融合
    
    # 获取最终检测区域
    final_contour = None
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        final_contour = max(contours, key=lambda cnt: cv2.contourArea(cnt))
    
    # 特征匹配（仅ROI区域）
    kp1, des1 = orb.detectAndCompute(base_enhanced, None)
    kp2, des2 = orb.detectAndCompute(current_enhanced, None)
    
    match_ratio = 0
    if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
        matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
        good = [m for m,n in matches if m.distance < 0.75*n.distance]
        match_ratio = len(good) / min(len(des1), len(des2))
        
    # 光流分析（仅ROI区域）
    if last_frame is None:
        last_frame = gray_roi.copy()
    flow = cv2.calcOpticalFlowFarneback(last_frame, gray_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    last_frame = gray_roi.copy()
    
    # 综合判断
    fg_ratio = np.sum(combined_mask)/255 / (w*h) if combined_mask.any() else 0
    motion_level = np.mean(np.abs(flow)) if flow is not None else 0
    
    condition = (
        fg_ratio > FG_RATIO_THRESH
        and match_ratio < MATCH_RATIO_THRESH
        and motion_level > MOTION_THRESH
    )

    # 可视化结果（保持原始图像尺寸）
    result = current_img.copy()
    roi_visual = cv2.addWeighted(current_enhanced, 0.3, cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR), 0.7, 0)
    result[y:y+h, x:x+w] = roi_visual
    # 可视化新增颜色检测结果
    result[y:y+h, x:x+w] = cv2.addWeighted(roi_visual, 0.7, cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    
    # 仅绘制最大检测框
    if final_contour is not None:
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(final_contour)
        start_point = (x + x_cnt, y + y_cnt)
        end_point = (x + x_cnt + w_cnt, y + y_cnt + h_cnt)
        cv2.rectangle(result, start_point, end_point, (0, 0, 255), 3)
        area_ratio = cv2.contourArea(final_contour)/(w*h)
        cv2.putText(result, f"{area_ratio:.1%}", (x + x_cnt + w_cnt - 120, y + y_cnt + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # 状态显示
    cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(result, f"FG: {fg_ratio:.1%}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(result, f"Match: {match_ratio:.1%}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(result, f"Motion: {motion_level:.1f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(result, f"Area: {fg_ratio:.1%}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # 创建全尺寸掩码
    combined_mask_full = np.zeros(current_img.shape[:2], dtype=np.uint8)
    combined_mask_full[y:y+h, x:x+w] = combined_mask
    
    return condition, result, combined_mask_full


def select_video():
    """选择视频文件"""
    global cap, roi_points, selecting_roi
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(initialdir="../data/video", title="选择视频文件", filetypes=[("视频文件", "*.mp4;*.avi")])
    return file_path

if __name__ == "__main__":
    '''
    使用说明：
    #################################################
    检测说明：
    1. 启动后自动进入预览模式，视频自动播放
    2. 按空格键可暂停视频，再次按空格继续
    3. 暂停时按回车确认起始帧，进入检测模式
    4. 检测模式下按Q键可提前终止检测
    5. 结果视频自动保存至../data/output文件夹
    
    标定说明：
    1. 在检测模式下，按住鼠标左键拖动选择矩形区域
    2. 松开鼠标完成区域选择
    3. 按 R 键确认ROI设置
    4. 按回车键确认起始帧并开始检测
    '''
    # 初始化视频窗口
    cv2.namedWindow('Video Preview')
    cv2.setMouseCallback('Video Preview', select_roi)
    
    # 初始化视频源
    video_path = select_video()
    # 如果未选择视频文件，提示并退出
    if not video_path:
        print("未选择视频文件")
        exit()
    # 初始化视频捕获
    cap = cv2.VideoCapture(video_path)
    
    # ================= 视频预览阶段 =================
    preview_mode = True
    start_frame = 0
    cv2.namedWindow('Video Preview')
    print("操作说明：")
    print("[空格键] 暂停/继续 | [回车键] 确认起始帧 | [Q键] 退出预览")

    while preview_mode and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 显示带帧号的预览画面
        preview = frame.copy()
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.putText(preview, f"The current frame: {current_frame}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(preview, "Preview mode - Browse videos to select a starting point", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow('Video Preview', preview)
        
         # 添加ROI标定提示
        if len(roi_points) == 1 and selecting_roi:
            cv2.putText(preview, "正在标定ROI... 拖动鼠标选择区域", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
        # 绘制实时ROI框
        if len(roi_points) == 2:
            x1, y1 = roi_points[0]
            x2, y2 = roi_points[1]
            cv2.rectangle(preview, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(preview, "按 R 确认ROI区域", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # 键盘控制逻辑
        key = cv2.waitKey(25)
        if key == ord(' '):  # 空格键暂停
            while True:
                key_pause = cv2.waitKey(0)
                if key_pause == ord('r') and len(roi_points) == 2:  # 确认ROI
                    x1, y1 = roi_points[0]
                    x2, y2 = roi_points[1]
                    roi = (min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1))
                    print(f"ROI已设置: {roi}")
                    break
                elif key_pause == 13:      # 回车确认起始帧
                    start_frame = current_frame
                    preview_mode = False
                    break
                elif key_pause == ord('q'):  # 退出程序
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
        elif key == 13:  # 直接按回车从头开始
            preview_mode = False
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # ================= 检测初始化阶段 =================
    # 重新初始化视频捕获到起始位置
    cap.release()
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 获取基准帧（使用用户选择的起始帧）
    ret, base_frame = cap.read()
    if not ret:
        print("错误：无法读取起始帧")
        exit()

    # 初始化检测参数
    roi = (300, 100, 600, 500)  # 检测区域坐标(x,y,w,h)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (base_frame.shape[1], base_frame.shape[0])
    
    # 初始化视频保存器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../data/output.avi', fourcc, fps, frame_size)
    
    # 重置背景减除模型
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)

    # ================= 主检测循环 =================
    print(f"开始检测，起始帧：{start_frame}")
    cv2.namedWindow('Detection Preview')
    
    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        # 执行异常检测
        is_anomaly, result_img, _ = dynamic_difference_detection(
            base_frame, current_frame, roi
        )

        # 异常标注
        if is_anomaly:
            print(f"帧 {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} 检测到异常！")
            cv2.putText(result_img, "ALERT!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 显示并保存结果
        cv2.imshow('Detection Preview', result_img)
        out.write(result_img)

        # 退出控制
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("检测完成，结果已保存至 ../data/output/output.avi")