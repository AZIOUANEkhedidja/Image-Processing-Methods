import os
import cv2
import tkinter as tk
from tkinter import Label, Menu, messagebox, filedialog, SOLID
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_conversion_and_histogram import *  
from image_conversion_and_histogram_opencv import *

# ------------------------------------- Main Application Class -------------------------------------

class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.geometry("1300x400")
        
        self.path = None
        self.img = None
        
        self.menu_bar = Menu(self.master)
        self.create_widgets()
        self.create_menus()

    def create_widgets(self):
        self.img_label = Label(self.master)
        self.img_label.place(x=0, y=50)

        button = tk.Button(self.master, text="Choose an Image", command=self.open_file)
        button.place(x=0, y=0)

    def create_menus(self):
        conversion_menu = Menu(self.menu_bar, tearoff=0)
        self.add_conversion_menu_items(conversion_menu)

        histogram_menu = Menu(self.menu_bar, tearoff=0)
        self.add_histogram_menu_items(histogram_menu)

        self.menu_bar.add_cascade(label="Conversion", menu=conversion_menu)
        self.menu_bar.add_cascade(label="Histogram", menu=histogram_menu)
        self.master.config(menu=self.menu_bar)

    def add_conversion_menu_items(self, menu):
        conversion_functions = [
            ("RGB to HSV", lambda: self.check_image_and_process(self.display_image, 'RGB', [self.img, rgb_to_hsv(self.img), rgb_to_hsv_opencv(self.img)])),
            ("HSV to RGB", lambda: self.check_image_and_process(self.display_image, 'HSV', [rgb_to_hsv(self.img), hsv_to_rgb(rgb_to_hsv_opencv(self.img)), hsv_to_rgb_opencv(rgb_to_hsv_opencv(self.img))])),
            ("RGB to Gray", lambda: self.check_image_and_process(self.display_image, 'RGB', [self.img, rgb_to_gray(self.img), rgb_to_gray_opencv(self.img)])),
            ("Gray to RGB", lambda: self.check_image_and_process(self.display_image, 'GRAY', [rgb_to_gray(self.img), gray_to_rgb(rgb_to_gray(self.img)), gray_to_rgb_opencv(rgb_to_gray_opencv(self.img))])),
            ("RGB to Binary", lambda: self.check_image_and_process(self.display_image, 'RGB', [self.img, gray_to_binary(rgb_to_gray(self.img)), rgb_to_binary_opencv(self.img)])),
            ("RGB to YCbCr", lambda: self.check_image_and_process(self.display_image, 'RGB', [self.img, rgb_to_ycbcr(self.img), rgb_to_ycbcr_opencv(self.img)])),
            ("YCbCr to RGB", lambda: self.check_image_and_process(self.display_image, "YCbCr", [rgb_to_ycbcr(self.img), ycbcr_to_rgb(rgb_to_ycbcr(self.img)), ycbcr_to_rgb_opencv(rgb_to_ycbcr_opencv(self.img))])),
            ("YCbCr to Binary", lambda: self.check_image_and_process(self.display_image, "YCbCr", [rgb_to_ycbcr(self.img), gray_to_binary(rgb_to_gray(ycbcr_to_rgb(rgb_to_ycbcr(self.img)))), ycbcr_to_binary_opencv(rgb_to_ycbcr_opencv(self.img))])),
            ("HSV to Binary", lambda: self.check_image_and_process(self.display_image, "HSV", [rgb_to_hsv(self.img), gray_to_binary(rgb_to_gray(hsv_to_rgb(rgb_to_hsv(self.img)))), hsv_to_binary_opencv(rgb_to_hsv_opencv(self.img))]))
        ]
        for label, command in conversion_functions:
            menu.add_command(label=label, command=command)
        menu.add_separator()

    def add_histogram_menu_items(self, menu):
        histogram_functions = [
            ("Histogram", lambda: self.check_image_and_process(self.display_histogram, [histogram_method(self.path), histogram_method_opencv2(self.path)], ['Histogram Code', 'Histogram OpenCV'], 400)),
            ("Gray Histogram", lambda: self.check_image_and_process(self.display_histogram_ng, [histogram_method_ng(self.path), histogram_method_opencv2_ng(self.path)], ['Histogram Code', 'Histogram OpenCV'], 400, self.path)),
            ("Cumulative Histogram", lambda: self.check_image_and_process(self.display_histogram, [histogramme_cumule(self.path), histogramme_cumule_opencv(self.path)], ['Cumulative Histogram Code', 'Cumulative Histogram OpenCV'], 400)),
            ("Cumulative Gray Histogram", lambda: self.check_image_and_process(self.display_histogram_ng, [histogramme_cumule_ng(self.path), histogramme_cumule_opencv_ng(self.path)], ['Cumulative Histogram Code', 'Cumulative Histogram OpenCV'], 400, self.path))
        ]

        for label, command in histogram_functions:
            menu.add_command(label=label, command=command)
            menu.add_separator()


    def open_file(self):
        file_path = filedialog.askopenfilename(initialdir="./images", title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.path = os.path.relpath(file_path, start=os.getcwd())
            self.img = cv2.imread(file_path)
            img_1 = Image.open(file_path)
            img_tk = ImageTk.PhotoImage(img_1)
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk
            self.img_label.place(x=0, y=50)

    def check_image_and_process(self, algorithm_func, *args):
        if self.img is not None:
            algorithm_func(*args)
        else:
            messagebox.showerror("Error", "Please select an image first")

    def display_image(self, title, imgs):
        titles = ['CODE-MANUAL', 'OPEN-CV2']
        titles.insert(0, title)
        x, y = 0, 50
        for i in range(3):
            img = imgs[i]
            title = titles[i]
            self.save_image(img, f'images/{title}.png')
            img = Image.open(f'images/{title}.png')
            img = ImageTk.PhotoImage(img)
            panel = Label(self.master, image=img)
            title_label = Label(self.master, text=title, relief=SOLID, width=20)
            y = img.height() + 50
            title_label.place(x=x, y=y)
            panel.image = img
            panel.place(x=x, y=50)
            x += img.width()
        self.master.update()

    def display_histogram(self, rgb_tabs, title, x, path=None):
        for i in range(2):
            r_hist, g_hist, b_hist = rgb_tabs[i]
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.plot(r_hist, color="red", label="Red Channel")
            ax.plot(g_hist, color="green", label="Green Channel")
            ax.plot(b_hist, color="blue", label="Blue Channel")
            ax.set_title(f"Histogram for {title[i]}")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")
            ax.legend()
            canvas = FigureCanvasTkAgg(fig, master=self.master)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.place(x=x, y=50)
            if i == 0:
                figure_width = fig.get_size_inches()[0] * fig.dpi
            x += figure_width + 70
            plt.close(fig)

        self.master.update()

    def display_histogram_ng(self, rgb_tabs, title, x, path=None):
        for i in range(2):
            r_hist = rgb_tabs[i]
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.plot(r_hist, color="black")
            ax.set_title(f"Histogram for {title[i]}")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")
            canvas = FigureCanvasTkAgg(fig, master=self.master)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.place(x=x, y=50)
            if i == 0:
                figure_width = fig.get_size_inches()[0] * fig.dpi
            x += figure_width + 70
            plt.close(fig)

        self.master.update()

    def save_image(self, img, name):
        cv2.imwrite(name, img)

# ------------------------------------- Main Functionality -------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
