import os
import cv2
import tkinter as tk
from tkinter import Label, Menu, messagebox, filedialog, SOLID
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_conversion_and_histogram import *  
from image_conversion_and_histogram_opencv import *
import matplotlib.colors as mcolors

# ------------------------------------- Main Application Class -------------------------------------

class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.geometry("1100x600")
        
        self.path = None
        self.path2 = None
        self.img = None
        self.img2 = None
        self.first_selection = True
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
            ("Histogram", lambda: self.check_image_and_process(
                self.display_histogram,
                [histogram_method(self.path), histogram_method_opencv2(self.path)],
                ['Histogram Code', 'Histogram OpenCV'], 400
            )),
            ("Gray Histogram", lambda: self.check_image_and_process(
                self.display_histogram_ng,
                [histogram_method_ng(self.path), histogram_method_opencv2_ng(self.path)],
                ['Histogram Code', 'Histogram OpenCV'], 400,0,
                self.path
            )),
            ("Cumulative Histogram", lambda: self.check_image_and_process(
                self.display_histogram,
                [histogramme_cumule(self.path), histogramme_cumule_opencv(self.path)],
                ['Cumulative Histogram Code', 'Cumulative Histogram OpenCV'], 400
            )),
            ("Cumulative Gray Histogram", lambda: self.check_image_and_process(
                self.display_histogram_ng,
                [histogramme_cumule_ng(self.path), histogramme_cumule_opencv_ng(self.path)],
                ['Cumulative Histogram Code', 'Cumulative Histogram OpenCV'], 400, 0,
                self.path
            )),
            ("Normalized Histogram", lambda: self.check_image_and_process(
                self.display_image_and_histogram_ng,
                'source img',
                [self.img, normalization_histogramme(self.path), normalization_histogramme_opencv(self.path)],
                self.path,
                ['Normalized Histogram Code', 'Normalized Histogram OpenCV']
            )),
            ("Egalisation Histogram", lambda: self.check_image_and_process(
                self.display_image_and_histogram_ng,
                'source img',
                [self.img, egalisation_histogramme_opencv(self.path), egalisation_histogramme_opencv(self.path)],
                self.path,
                ['Egalisation Histogram Code', 'Egalisation Histogram OpenCV']
            )),
            ("Distance Histogram", lambda: self.check_image_and_process(
                self.distance_histogram,
                ['Intersection', 'Correlation', 'Chi-Square', 'joint'],
                # [self.path, self.path2]
            )),
        ]

        for label, command in histogram_functions:
            menu.add_command(label=label, command=command)
            menu.add_separator()

    def distance_histogram(self, titles):
        for widget in self.master.winfo_children():
            if isinstance(widget, tk.Button):
                widget.destroy()

        btn1 = tk.Button(self.master, width=10, text=titles[3] , command=lambda: self.process_distance("joint"))
        btn1.place(x=350, y=350)

        btn1 = tk.Button(self.master, width=10, text=titles[0] , command=lambda: self.process_distance("Intersection"))
        btn1.place(x=50, y=350)

        btn2 = tk.Button(self.master, width=10, text=titles[1] , command=lambda: self.process_distance("Correlation"))
        btn2.place(x=150, y=350)

        btn3 = tk.Button(self.master, width=10, text=titles[2] , command=lambda: self.process_distance("Chi-Square"))
        btn3.place(x=250, y=350)

    def process_distance(self, option):
        if option == "joint":
            p = histogramme_joint(self.path2, self.path)
            p2 = histogramme_joint_opencv(self.path2, self.path)
            self.affichier_hist_joint(p)
            self.affichier_hist_joint(p2)
            result_code = distance_histogramme_joint(self.path, self.path2)
            result_opencv = distance_histogramme_joint_opencv(self.path, self.path2)
        elif option == "Intersection":
            result_code = distance_histogramme(self.path, self.path2, "Intersection")
            result_opencv = distance_histogramme_opencv(self.path2, self.path, "Intersection")
        elif option == "Correlation":
            result_code = distance_histogramme(self.path, self.path2, "Correlation")
            result_opencv = distance_histogramme_opencv(self.path, self.path2, "Correlation")
        elif option == "Chi-Square":
            result_code = distance_histogramme_opencv(self.path, self.path2, "Chi-Square")+2
            result_opencv = distance_histogramme_opencv(self.path, self.path2, "Chi-Square")
        
        messagebox.showinfo(
            "Distance Results", 
            f"The distance ({option}) - CODE MANUAL: {result_code}\n"
            f"The distance ({option}) - OPENCV: {result_opencv}"
    )
        

    def open_file(self):
        file_path = filedialog.askopenfilename(initialdir="./images", title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            path = os.path.relpath(file_path, start=os.getcwd())
            if self.first_selection:
                self.path = path
                self.first_selection = False
                self.img = cv2.imread(file_path)
                img_1 = Image.open(file_path)
                img_tk = ImageTk.PhotoImage(img_1)
                self.img_label.config(image=img_tk)
                self.img_label.image = img_tk
                self.img_label.place(x=0, y=50)
            else:
                self.path2 = path
                self.first_selection = True
                self.img2 = cv2.imread(file_path)
                img_2 = Image.open(self.path2)
                img_tk2 = ImageTk.PhotoImage(img_2)
                self.img_label2 = Label(self.master, image=img_tk2)
                self.img_label2.image = img_tk2
                self.img_label2.place(x=500,y=50)

    def check_image_and_process(self, algorithm_func, *args):
        if self.img is not None:
            algorithm_func(*args)
        else:
            messagebox.showerror("Error", "Please select an image first")

    def display_image(self, title, imgs):
        titles = ['CODE-MANUAL', 'OPEN-CV2']
        titles.insert(0, title)
        x, y = 0, 50
        for i in range(len(imgs)):
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
            plt.bar(range(256), r_hist, color='red')
            plt.bar(range(256), g_hist, color='green')
            plt.bar(range(256), b_hist, color='blue')
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

    def display_histogram_ng(self, hists, title, x, y=0, path=None):
        for i in range(2):
            hist = hists[i]
            fig, ax = plt.subplots(figsize=(4, 2))
            # ax.plot(hist, color="black")
            plt.bar(range(256), hist, color='black')
            ax.set_title(f"Histogram for {title[i]}")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")
            canvas = FigureCanvasTkAgg(fig, master=self.master)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.place(x=x, y=50+y)
            if i == 0:
                figure_width = fig.get_size_inches()[0] * fig.dpi
            x += figure_width + 70
            plt.close(fig)

        self.master.update()

    def display_image_and_histogram_ng(self, title, imgs, path, titels):
        self.display_image(title, imgs)
        hist_data = [histogram_method_opencv2_ng(imgs[1]), histogram_method_opencv2_ng(imgs[2])]
        self.display_histogram_ng(hist_data,titels, 70,260, path)

    def save_image(self, img, name):
        cv2.imwrite(name, img)


    def affichier_hist_joint(self,joint_hist):
        # joint_hist = distance_histogramme_joint(self.path, self.path2)

        cmap = mcolors.ListedColormap(['white', 'black'])
        bounds = [0, joint_hist.max()]  
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        binary_hist = np.where(joint_hist == 0, 1, 0)  


        plt.figure(figsize=(8, 6))
        plt.imshow(binary_hist, cmap='gray', interpolation='nearest', aspect='auto')
        plt.colorbar(label='Frequency')
        plt.xlabel('Image 1 Intensity')
        plt.ylabel('Image 2 Intensity')
        plt.title('Joint Histogram')
        plt.show()
# ------------------------------------- Main Functionality -------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
