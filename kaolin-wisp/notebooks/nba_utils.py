import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def create_vid(imgs, path, W, H):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 10, (W,H))
    for i in range(len(imgs)):
        img = imgs[i]
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)

def animate_plot(xs, ys, path, xlabel = "x", ylabel = "y", title = "plot"):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)

    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = xs[:i]
        y = ys[:i]
        line.set_data(x, y)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(xs), interval=100, blit=True)

    anim.save(path, writer='ffmpeg')

def merge_horizontal(video1_path, video2_path, out_path):
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

        
    # Get video properties
    fps = int(video1.get(cv2.CAP_PROP_FPS))
    frame_width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_height_2 = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width_2 = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))

    resize_ratio = frame_height / frame_height_2

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(out_path, fourcc, fps, (frame_width + int(frame_width_2 * resize_ratio) , frame_height))

    # Loop through the frames and merge them side by side
    for _ in range(num_frames):
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        if ret1 and ret2:
            if resize_ratio != 1.0:
                frame2 = cv2.resize(frame2, (0, 0), fx = resize_ratio, fy = resize_ratio)

            merged_frame = cv2.hconcat([frame1, frame2])
            output_video.write(merged_frame)
        else:
            break

    video1.release()
    video2.release()
    output_video.release()

    cv2.destroyAllWindows()

def uniform_spherical_volume_sample(n_samples, r1, r2):
    """
    Uniformly sample n_samples points in a spherical volume from radius r1 to r2
    """
    u = np.random.uniform(np.power(r1, 3), np.power(r2, 3), n_samples)
    r = np.power(u, 1/3)
    costheta = np.random.uniform(-1, 1, n_samples)
    theta = np.arccos(costheta)
    phi = np.random.uniform(0, 2*np.pi, n_samples)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack([x, y, z], axis = 1)
def ellipsoid_volume_sample(n_samples, r1x, r2x, r1y, r2y, r1z, r2z):
    """
    Uniformly sample n_samples points in a spherical volume from radius r1 to r2
    """
    rx = np.power(np.random.uniform(np.power(r1x, 3), np.power(r2x, 3), n_samples), 1/3)
    ry = np.power(np.random.uniform(np.power(r1y, 3), np.power(r2y, 3), n_samples), 1/3)
    rz = np.power(np.random.uniform(np.power(r1z, 3), np.power(r2z, 3), n_samples), 1/3)

    costheta = np.random.uniform(-1, 1, n_samples)
    theta = np.arccos(costheta)
    phi = np.random.uniform(0, 2*np.pi, n_samples)

    x = rx * np.sin(theta) * np.cos(phi)
    y = ry * np.sin(theta) * np.sin(phi)
    z = rz * np.cos(theta)

    return np.stack([x, y, z], axis = 1)


# def avg_sph_dist(pts, pt2, center):
#     dist_sum = 0
#     pt2 = (pt2-center) / np.linalg.norm(pt2-center)
#     for pt in pts:
#         pt = (pt-center) / np.linalg.norm(pt-center)
#         dist_sum += np.linalg.norm(pt - pt2)
#     return dist_sum / len(pts)

def avg_sph_dist(pts, pt2):
    dist_sum = 0
    pt2 = pt2 / np.linalg.norm(pt2)
    for pt in pts:
        pt = pt / np.linalg.norm(pt)
        dist_sum += np.linalg.norm(pt - pt2)
    return dist_sum / len(pts)
