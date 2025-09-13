import os
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageSequence
from scipy.spatial import ConvexHull
from numpy.linalg import eig
import cv2


class VideoAnnotationGUI:
    def __init__(self, master, frames, filepath):
        self.master = master
        self.frames = frames
        self.filepath = filepath
        self.total_frames = len(frames)
        self.current_frame_index = 0
        self.contour_locations = []
        self.current_clicks = []

        self.create_widgets()
        self.show_frame()

    def create_widgets(self):
        """Initialize GUI components"""
        self.frame_label = tk.Label(self.master)
        self.frame_label.pack()
        
        self.index_label = tk.Label(self.master, 
                                  text=f"Frame {self.current_frame_index + 1}/{self.total_frames}")
        self.index_label.pack()
        
        self.canvas = tk.Canvas(self.master, 
                              width=self.frames[0].shape[1], 
                              height=self.frames[0].shape[0])
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=10)
        
        nav_frame = tk.Frame(control_frame)
        nav_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(nav_frame, text="Jump to:").pack(side=tk.LEFT)
        self.jump_entry = tk.Entry(nav_frame, width=5)
        self.jump_entry.pack(side=tk.LEFT)
        tk.Button(nav_frame, text="Go", command=self.jump_to_frame).pack(side=tk.LEFT)
        
        tk.Button(nav_frame, text="◀ Last", command=self.show_last_frame).pack(side=tk.LEFT)
        tk.Button(nav_frame, text="Next ▶", command=self.show_next_frame).pack(side=tk.LEFT)
        
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Redo", command=self.redo_last_contour).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Save", command=self.save_annotations).pack(side=tk.LEFT)

    def show_frame(self):
        """Reset canvas and display current frame"""
        self.current_clicks = []
        self.canvas.delete("click_points")
        self.canvas.delete("contour")
        
        frame = self.frames[self.current_frame_index]
        frame_uint8 = (255 * (frame - frame.min()) / (frame.max() - frame.min())).astype(np.uint8)
        img = Image.fromarray(frame_uint8)
        self.img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        self.index_label.config(text=f"Frame {self.current_frame_index + 1}/{self.total_frames}")

    def show_last_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.show_frame()

    def show_next_frame(self):
        if self.current_frame_index < self.total_frames - 1:
            self.current_frame_index += 1
            self.show_frame()

    def jump_to_frame(self):
        try:
            frame_num = int(self.jump_entry.get())
            if 1 <= frame_num <= self.total_frames:
                self.current_frame_index = frame_num - 1
                self.show_frame()
        except ValueError:
            pass

    def redo_last_contour(self):
        if self.contour_locations:
            self.contour_locations.pop()
            self.show_frame()

    def on_canvas_click(self, event):
        """Handle four-point click input"""
        if len(self.current_clicks) >= 4:
            return

        x, y = event.x, event.y
        self.current_clicks.append((x, y))
        self.canvas.create_oval(x-3, y-3, x+3, y+3, 
                               fill='red', tags="click_points")

        if len(self.current_clicks) == 4:
            self.create_rotated_ellipse()
            self.master.after(500, self.show_next_frame)

    def create_rotated_ellipse(self):
        """Calculate rotated ellipse parameters using PCA"""
        try:
            points = np.array(self.current_clicks)
            
            # Get convex hull of points
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # Calculate PCA components
            centroid = np.mean(hull_points, axis=0)
            centered = hull_points - centroid
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = eig(cov)
            
            # Sort by eigenvalue magnitude
            order = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:, order]
            
            # Calculate axis lengths and angle
            projected = np.dot(centered, eigenvectors)
            max_vals = np.max(projected, axis=0)
            min_vals = np.min(projected, axis=0)
            lengths = (max_vals - min_vals)
            
            angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
            
            # Store parameters correctly as (frame_idx, (cx, cy, w, h, angle))
            self.contour_locations.append(
                (self.current_frame_index, 
                 (centroid[0], centroid[1],
                  lengths[0], lengths[1],
                  angle))
            )
            
            self.draw_rotated_ellipse(centroid, lengths, angle)

        except Exception as e:
            print(f"Error creating ellipse: {str(e)}")
            self.current_clicks = []
            self.canvas.delete("click_points")

    def draw_rotated_ellipse(self, center, lengths, angle):
        """Draw rotated ellipse on canvas"""
        self.canvas.delete("contour")
        cx, cy = center
        a, b = lengths[0]/2, lengths[1]/2
        
        # Generate ellipse points
        theta = np.linspace(0, 2*np.pi, 50)
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        
        # Rotation matrix
        rad_angle = np.radians(angle)
        rot = np.array([
            [np.cos(rad_angle), -np.sin(rad_angle)],
            [np.sin(rad_angle), np.cos(rad_angle)]
        ])
        
        # Transform points
        points = np.dot(rot, np.vstack((x, y)))
        points[0] += cx
        points[1] += cy
        
        # Create polygon
        self.canvas.create_polygon(
            *zip(points[0], points[1]),
            outline='yellow', fill='', width=2,
            smooth=True, tags="contour"
        )

    def save_annotations(self):
        """Save ellipse parameters to file"""
        with open(self.filepath, 'w') as f:
            for frame_idx, params in self.contour_locations:
                line = (f"{frame_idx},{params[0]:.2f},{params[1]:.2f},"
                        f"{params[2]:.2f},{params[3]:.2f},{params[4]:.2f}\n")
                f.write(line)
        print(f"Ellipse data saved to {self.filepath}")

# ------------------------
def load_tif_frames(tif_path, sample_interval=1):
    """Load frames from a multi-page TIFF"""
    img = Image.open(tif_path)
    frames = []
    for i, page in enumerate(ImageSequence.Iterator(img)):
        if i % sample_interval == 0:
            arr = np.array(page).astype(np.float32)
            # normalize
            normalized = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            frames.append(normalized)
    return frames

def load_avi_frames(video_path, sample_interval=4):
    """Load frames from AVI video with optional sampling"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            normalized = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
            frames.append(normalized)
            
        frame_count += 1
    
    cap.release()
    return frames

def generate_filepath(base_filename, directory):
    """Generate unique save path by incrementing suffix"""
    os.makedirs(directory, exist_ok=True)
    suffix = 0
    while True:
        filename = f"{base_filename}_{suffix}.txt"
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            return filepath
        suffix += 1

# ------------------------
def main():
    root = tk.Tk()
    root.title("Rotated Ellipse Annotation")

    directory = 'D:/hippo data/bio_image/day14'
    base_filename = './T_4x1_anno'
    filepath = generate_filepath(base_filename, directory)
    
    # Update to your TIFF file
    video_path = 'D:/hippo data/bio_image/day14/day14.tif'
    
    try:
        if video_path.lower().endswith((".tif", ".tiff")):
            frames = load_tif_frames(video_path, sample_interval=1)
        else:
            frames = load_avi_frames(video_path, sample_interval=10)
        print(f"Loaded {len(frames)} frames from {video_path}")
    except Exception as e:
        print(f"Error loading frames: {str(e)}")
        return
    
    VideoAnnotationGUI(root, frames, filepath)
    root.mainloop()

if __name__ == "__main__":
    main()
