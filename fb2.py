import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import find_peaks, butter, filtfilt

# 全局设置字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18

def select_roi(frame):
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select ROI", frame)
    cv2.destroyAllWindows()
    return roi

cap = cv2.VideoCapture("NeckDeformationClip.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("optical_flow_fb2.mp4", fourcc, fps, (width, height))

ret, prev_frame = cap.read()
roi = select_roi(prev_frame)
x, y, w, h = map(int, roi)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

signals = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # 计算密集光流
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=5, poly_n=5, poly_sigma=1.2, flags=0
    )

    # 提取运动信息
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_magnitude = np.mean(magnitude)
    signals.append(avg_magnitude)

    # 可视化光流方向
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    # 将ROI区域的光流映射回原图坐标
    hsv_roi = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_roi[..., 0] = angle * 180 / np.pi / 2
    hsv_roi[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[y:y + h, x:x + w] = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)

    overlay = cv2.addWeighted(frame, 0.7, hsv, 0.3, 0)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
    out.write(overlay)

    prev_gray = curr_gray.copy()

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
plt.title(f'Farneback Optical Flow - Heart Rate: {heart_rate:.1f} BPM')
plt.legend(prop={'size': 16}, frameon=False, loc='best')
plt.savefig('optical_flow_fb2.png', dpi=600)
plt.show()