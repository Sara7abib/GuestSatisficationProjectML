import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import joblib
import os

# Load your trained ML model
MODEL_PATH = "your_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    print(f"Error loading model: {e}")

# Define colors for a modern palette
BG_COLOR = "#f9f9f9"
NAV_BG_COLOR = "#1e88e5"
NAV_BTN_BG = "#1565c0"
NAV_BTN_HOVER_BG = "#0d47a1"
TEXT_COLOR = "#212121"
ACCENT_COLOR = "#f57c00"
LABEL_COLOR = "#424242"
SEPARATOR_COLOR = "#e0e0e0"
BUTTON_BG = "#42a5f5"
BUTTON_HOVER_BG = "#1e88e5"
BUTTON_FG = "white"


class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Project GUI")
        self.root.geometry("950x650")
        self.root.configure(bg=BG_COLOR)

        # Navigation bar
        nav_frame = tk.Frame(root, bg=NAV_BG_COLOR)
        nav_frame.pack(side="top", fill="x")

        self.nav_buttons = []
        btn_info = [("About", self.show_about), ("Plots", self.show_plots), ("Predict", self.show_predict)]
        for text, cmd in btn_info:
            btn = tk.Button(nav_frame, text=text, width=15, bg=NAV_BTN_BG, fg="white",
                            font=("Segoe UI", 11, "bold"), bd=0, relief="flat",
                            activebackground=NAV_BTN_HOVER_BG, activeforeground="white",
                            command=cmd, cursor="hand2")
            btn.pack(side="left", padx=8, pady=10)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=NAV_BTN_HOVER_BG))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=NAV_BTN_BG))
            self.nav_buttons.append(btn)

        # Scrollable content area
        self.canvas = tk.Canvas(root, bg=BG_COLOR, highlightthickness=0)
        self.scroll_y = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.frame = tk.Frame(self.canvas, bg=BG_COLOR)

        self.frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y.pack(side="right", fill="y")

        # Store images to prevent garbage collection
        self.images = []

        # Variables to store CSV data and filepath
        self.csv_data = None
        self.csv_path = None

        self.show_about()  # Default page on start

    def show_about(self):
        self.clear_frame()

        tk.Label(
            self.frame,
            text="📌 About the Project",
            font=("Segoe UI", 20, "bold"),
            bg=BG_COLOR,
            fg=NAV_BG_COLOR
        ).pack(padx=20, pady=(20, 12), anchor="w")

        description = (
            "This project investigates the key factors that influence guest satisfaction in short-term rental stays, "
            "with a specific focus on Airbnb listings. By leveraging a diverse dataset that includes variables such as "
            "property amenities, host responsiveness, pricing strategies, listing descriptions, and guest reviews, "
            "we aim to uncover what truly impacts the overall guest experience.\n\n"

            "The core objective is to build robust machine learning models that can accurately predict guest satisfaction levels. "
            "We apply data preprocessing, feature engineering, and multiple classification algorithms to identify significant predictors "
            "and evaluate model performance.\n\n"

            "This analysis not only aids in understanding the dynamics of the short-term rental market but also provides practical "
            "recommendations for hosts and platforms. Insights from the project can be used to improve service quality, highlight "
            "critical listing attributes, enhance guest communication, and ultimately boost customer satisfaction and retention."
        )

        tk.Label(
            self.frame,
            text=description,
            justify="left",
            bg=BG_COLOR,
            wraplength=880,
            font=("Segoe UI", 13),
            fg=LABEL_COLOR
        ).pack(padx=25, pady=(0, 30), anchor="w")

        tk.Label(
            self.frame,
            text="👥 Team Members",
            font=("Segoe UI", 16, "bold"),
            bg=BG_COLOR,
            fg=ACCENT_COLOR
        ).pack(padx=20, pady=(10, 15), anchor="w")

        members = [
            ("Dareen Housam Eldin", "2022170141"),
            ("Rasha AbdElkhalik", "2022170151"),
            ("Rahma Ahmed Ahmed", "2022170148"),
            ("Engy Abdelmalak", "2022170078"),
            ("Sara Yasser Habib", "2022170181")
        ]

        for name, student_id in members:
            container = tk.Frame(self.frame, bg=BG_COLOR)
            container.pack(padx=30, pady=4, anchor="w")

            tk.Label(container, text=name, font=("Segoe UI", 13, "bold"), fg="#27ae60", bg=BG_COLOR).pack(side="left")
            tk.Label(container, text=f" — {student_id}", font=("Segoe UI", 13), fg="#2980b9", bg=BG_COLOR).pack(side="left")

    def show_plots(self):
        self.clear_frame()
        tk.Label(self.frame, text="📈 Data Plots", font=("Segoe UI", 18, "bold"), bg=BG_COLOR, fg=NAV_BG_COLOR).pack(pady=(20, 15))

        plots = [
            ("plot1.png", "Distribution of Guest Satisfaction."),
            ("plot2.png", "Proportion of Guest Satisfaction by Room Type."),
            ("plot3.png", "Proportion of Guest Satisfaction by Host Superhost Status."),
            ("plot4.png", "host_listings_count vs guest_satisfaction."),
            ("plot5.png", "Geographic Distribution of Listings Colored by guest_satisfaction."),
            ("plot6.png", "Top 15 Correlations with 'guest_satisfaction'."),
            ("plot7.png", "Best Random Forest Confusion Matrix."),
            ("plot8.png", "Best Gradient Boosting Confusion Matrix."),
            ("plot9.png", "Best KNN Confusion Matrix (with Scaling).")

        ]

        for plot_path, description in plots:
            if os.path.exists(plot_path):
                try:
                    img = Image.open(plot_path)
                    img = img.resize((820, 470), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.images.append(photo)

                    tk.Label(self.frame, image=photo, bg=BG_COLOR, bd=0).pack(pady=(15, 6))
                    tk.Label(self.frame, text=description, bg=BG_COLOR, font=("Segoe UI", 12), wraplength=880,
                             justify="left", fg=TEXT_COLOR).pack(pady=(0, 15))

                    separator = tk.Frame(self.frame, height=2, bd=0, bg=SEPARATOR_COLOR)
                    separator.pack(fill="x", padx=30, pady=15)

                except Exception as e:
                    tk.Label(self.frame, text=f"⚠️ Error loading {plot_path}: {e}", bg=BG_COLOR, fg="red").pack(pady=5)
            else:
                tk.Label(self.frame, text=f"⚠️ Plot not found: {plot_path}", bg=BG_COLOR, fg="red").pack(pady=5)

    def show_predict(self):
        self.clear_frame()
        self.csv_data = None
        self.csv_path = None

        browse_btn = tk.Button(self.frame, text="Browse CSV File", font=("Segoe UI", 14, "bold"),
                               bg=BUTTON_BG, fg=BUTTON_FG, activebackground=BUTTON_HOVER_BG, activeforeground=BUTTON_FG,
                               cursor="hand2", command=self.browse_csv)
        browse_btn.pack(pady=(25, 20), ipadx=10, ipady=7)

        # Text widget to show CSV contents or predictions
        self.result_text = tk.Text(self.frame, width=110, height=22, bg="white", fg=TEXT_COLOR, font=("Segoe UI", 11),
                                   bd=2, relief="sunken")
        self.result_text.pack(pady=(0, 20), padx=25)
        self.result_text.config(state="disabled")

        # Predict button below text widget
        self.predict_btn = tk.Button(self.frame, text="Predict", font=("Segoe UI", 14, "bold"),
                                     bg=BUTTON_BG, fg=BUTTON_FG, activebackground=BUTTON_HOVER_BG, activeforeground=BUTTON_FG,
                                     cursor="hand2", state="disabled", command=self.run_prediction)
        self.predict_btn.pack(pady=(0, 30), ipadx=10, ipady=7)

    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            self.csv_data = pd.read_csv(file_path)
            self.csv_path = file_path

            self.result_text.config(state="normal")
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"📄 Loaded CSV: {os.path.basename(file_path)}\n\n")
            self.result_text.insert(tk.END, self.csv_data.head(20).to_string())
            self.result_text.config(state="disabled")

            # Enable predict button only if model is loaded
            if model_loaded:
                self.predict_btn.config(state="normal")
            else:
                self.predict_btn.config(state="disabled")
                messagebox.showinfo("Model Not Loaded", "Model not loaded.\nPrediction not available.\nShowing CSV content only.")

        except Exception as e:
            messagebox.showerror("File Error", f"Failed to load CSV file.\n\nError: {e}")

    def run_prediction(self):
        if self.csv_data is None:
            messagebox.showwarning("No CSV", "Please load a CSV file first.")
            return

        try:
            predictions = model.predict(self.csv_data)

            self.result_text.config(state="normal")
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "✅ Predictions:\n\n")
            for i, pred in enumerate(predictions):
                self.result_text.insert(tk.END, f"Row {i + 1}: {pred}\n")
            self.result_text.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Could not perform prediction.\n\nError: {e}")

    def clear_frame(self):
        for widget in self.frame.winfo_children():
            widget.destroy()
        self.images.clear()


if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
