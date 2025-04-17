import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
from scipy.ndimage import median_filter

# 全局设置字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.titlesize'] = 20  # 子图标题字体大小
rcParams['axes.labelsize'] = 18  # 坐标轴标签字体大小
rcParams['xtick.labelsize'] = 18  # x 轴刻度字体大小
rcParams['ytick.labelsize'] = 18  # y 轴刻度字体大小

# Savitzky-Golay微分核 (7点三次一阶)
SGD_KERNEL = np.array([22, -67, -58, 0, 58, 67, -22]) / 252.0

def select_roi(frame):
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select ROI", frame)
    cv2.destroyAllWindows()
    return roi

def compute_gradient(image):
    grad_x = cv2.filter2D(image, -1, SGD_KERNEL.reshape(1, -1), borderType=cv2.BORDER_REPLICATE)
    grad_y = cv2.filter2D(image, -1, SGD_KERNEL.reshape(-1, 1), borderType=cv2.BORDER_REPLICATE)
    return grad_x + 1j * grad_y

def phase_correlation(grad1, grad2):
    # 归一化交叉功率谱
    fft1 = np.fft.fft2(grad1)
    fft2 = np.fft.fft2(grad2)
    cross_power = fft1 * np.conj(fft2)
    cross_power /= (np.abs(cross_power) + 1e-8)

    # 计算相关图并找峰值
    corr = np.fft.fftshift(np.fft.ifft2(cross_power).real)
    max_loc = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = grad1.shape
    return (max_loc[0] - h // 2, max_loc[1] - w // 2)

def subpixel_shift(img1, img2, init_shift):
    # Hann窗预处理
    hann = np.outer(np.hanning(img1.shape[0]), np.hanning(img1.shape[1]))
    img1_win = img1 * hann
    img2_win = img2 * hann

    # 计算相位差
    fft1 = np.fft.fft2(img1_win)
    fft2 = np.fft.fft2(img2_win)
    phase_diff = np.angle(fft1 * np.conj(fft2))

    # 中值滤波去噪
    phase_diff = median_filter(phase_diff, size=3)

    # 生成频率网格
    ky, kx = np.mgrid[-img1.shape[0] / 2:img1.shape[0] / 2,
             -img1.shape[1] / 2:img1.shape[1] / 2]
    kx = kx / img1.shape[1]
    ky = ky / img1.shape[0]

    # 选择中心区域(75%)
    mask = (np.abs(kx) < 0.75) & (np.abs(ky) < 0.75)
    A = np.vstack([kx[mask], ky[mask], np.ones(mask.sum())]).T
    b = phase_diff[mask].flatten()

    # 最小二乘求解相位梯度
    dx, dy, _ = np.linalg.lstsq(A, b, rcond=None)[0]
    return init_shift + (-dx / (2 * np.pi), -dy / (2 * np.pi))


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('subpixel.mp4', fourcc, fps, (width, height))
    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = select_roi(display_frame)
    x, y, w, h = map(int, roi)

    # 初始化参考帧
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w].astype(np.float32)
    prev_grad = compute_gradient(prev_gray)

    displacements = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 可视化当前帧
        vis_frame = frame.copy()

        # 仅处理ROI区域
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w].astype(np.float32)
        curr_grad = compute_gradient(curr_gray)

        # 计算位移
        init_shift = phase_correlation(prev_grad, curr_grad)
        final_shift = subpixel_shift(prev_gray, curr_gray, init_shift)
        displacement = np.linalg.norm(final_shift)
        displacements.append(displacement)

        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 绘制位移矢量
        center = (x + w // 2, y + h // 2)
        end_point = (int(center[0] + final_shift[1] * 20),  # 放大20倍
                     int(center[1] + final_shift[0] * 20))
        cv2.arrowedLine(vis_frame, center, end_point, (0, 0, 255), 2, tipLength=0.3)

        # 显示位移数值
        cv2.putText(vis_frame, f"DX: {final_shift[1]:.2f} px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"DY: {final_shift[0]:.2f} px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Total: {displacement:.2f} px", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # 写入视频帧
        out.write(vis_frame)

        prev_gray = curr_gray.copy()
        prev_grad = curr_grad

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return np.array(displacements), fps

def process_signals(signal, fps):
    nyq = 0.5 * fps
    b, a = butter(6, [0.5 / nyq, 4 / nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    peaks, _ = find_peaks(filtered, distance=fps * 0.5)
    heart_rate = len(peaks) / (len(signal) / fps) * 60

    plt.figure(figsize=(12, 6))
    time = np.arange(len(signal)) / fps
    plt.plot(time, signal, 'g', alpha=0.5, label='Raw Signal', markeredgewidth=1.5)
    plt.plot(time, filtered, label='Filtered Signal', markeredgewidth=2)
    plt.plot(time[peaks], filtered[peaks], 'rx', label='Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Motion Magnitude')
    plt.title(f'P-SG-GC - Heart Rate: {heart_rate:.1f} BPM')
    plt.legend(prop={'size': 16}, frameon=False, loc ='best')
    plt.savefig('subpixel.png', dpi=600)
    plt.show()

if __name__ == "__main__":
    video_path = "NeckDeformationClip.mp4"
    displacements, fps = analyze_video(video_path)
    process_signals(displacements, fps)
