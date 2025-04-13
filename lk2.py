import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from matplotlib import rcParams

# 全局设置字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18

def select_roi(frame):
    cv2.imshow("Select ROI", frame)
    roi = cv2.selectROI("Select ROI", frame)
    cv2.destroyAllWindows()
    return roi

cap = cv2.VideoCapture("NeckDeformationClip.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("optical_flow_lk2.mp4", fourcc, fps, (width, height))

ret, prev_frame = cap.read()
roi = select_roi(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))
x, y, w, h = map(int, roi)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

# 创建ROI mask
roi_mask = np.zeros_like(prev_gray)
roi_mask[:] = 255

# Shi-Tomasi角点检测参数
feature_params = dict(maxCorners=60,
                      qualityLevel=0.3,
                      minDistance=5,
                      blockSize=7,
                      mask=roi_mask)
p0 = cv2.goodFeaturesToTrack(prev_gray, **feature_params)

signals = []
mask = np.zeros_like(prev_frame)  # 光流可视化掩膜

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 裁剪当前帧到ROI区域
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # 计算稀疏光流
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)

    # 筛选有效跟踪点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 计算运动幅度信号
    motion_magnitude = np.mean(np.linalg.norm(good_new - good_old, axis=1))
    signals.append(motion_magnitude)

    # 绘制光流轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # 将ROI坐标转换回原始坐标系
        a, b = (new.ravel() + [x, y]).astype(int)
        c, d = (old.ravel() + [x, y]).astype(int)

        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    img = cv2.add(frame, mask)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    out.write(img)

    # 更新数据
    prev_gray = curr_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
out.release()

time = np.arange(len(signals)) / fps
nyquist = 0.5 * fps
b, a = butter(6, [0.5 / nyquist, 4 / nyquist], btype='band')
filtered = filtfilt(b, a, signals)
peaks, _ = find_peaks(filtered, distance=fps * 0.5)
heart_rate = len(peaks) / (len(signals) / fps) * 60

plt.figure(figsize=(12, 6))
plt.plot(time, signals, 'g', alpha=0.5, label='Raw Signal', markeredgewidth=1.5)
plt.plot(time, filtered, label='Filtered Signal', markeredgewidth=2)
plt.plot(time[peaks], filtered[peaks], 'rx', label='Peaks')
plt.xlabel('Time (s)')
plt.ylabel('Motion Magnitude')
plt.title(f'Lucas-Kanade Optical Flow - Heart Rate: {heart_rate:.1f} BPM')
plt.legend(prop={'size': 16}, frameon=False, loc='best')
plt.savefig('optical_flow_lk2.png', dpi=600)
plt.show()