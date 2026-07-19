import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import joblib
import os
import traceback
from test_script import load_and_clean_dataframe

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
            "This project delves into the critical factors influencing guest satisfaction in the short-term rental market, "
            "with a concentrated focus on Airbnb listings. Guest satisfaction is a key determinant of a property's success, directly impacting ratings, reviews, and future bookings. "
            "Understanding what drives high satisfaction can empower hosts to tailor their offerings for better customer experiences.\n\n"

            "We utilize a rich and diverse dataset that captures various aspects of Airbnb listings, including property amenities, location, host response rate and time, pricing strategies, "
            "length and tone of listing descriptions, and sentiment extracted from guest reviews. The data is cleaned, transformed, and enriched through comprehensive preprocessing and feature engineering steps "
            "to ensure the most informative inputs are used for modeling.\n\n"

            "The primary objective of the study is to predict guest satisfaction scores — typically reflected in the overall rating — using advanced regression techniques. "
            "We experiment with multiple regression models such as Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, and other ensemble methods. "
            "Model performance is evaluated using metrics like Mean Squared Error (MSE), and R² score to determine the most effective approach.\n\n"

            "Beyond the technical evaluation, we interpret model outputs to identify the most influential predictors of guest satisfaction. "
            "This insight provides valuable recommendations for Airbnb hosts and the platform itself — helping them prioritize improvements in service quality, pricing optimization, listing presentation, and guest interaction.\n\n"

            "Ultimately, this project combines data science and domain knowledge to offer actionable strategies that enhance the guest experience and promote long-term success in the competitive short-term rental market."
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
            ("plot1.png", "Distribution of Superhosts vs. Non-Superhosts."),
            ("plot2.png", "Proportion of Listings by Instant Bookable Status."),
            ("plot3.png", "Distribution of Review Scores Rating."),
            ("plot4.png", "Boxplot of Review Scores Rating."),
            ("plot5.png", "Average review_scores_rating by Host Superhost Status."),
            ("plot6.png", "Count of Listings by Cancellation Policy."),
            ("plot7.png", "Histogram of review_scores_rating."),
            ("plot8.png", "host_listings_count vs review_scores_rating."),
            ("plot9.png", "Pair Plot of Numeric Features and review_scores_rating."),
            ("plot10.png", "Geographic Distribution of Listings Colored by Review Scores Rating."),
            ("plot11.png", "Correlation Heatmap of Top 10 Features with review_scores_rating."),
            ("plot12.png", "MSE vs. Polynomial Degree"),
            ("plot13.png", "Ridge Regression: MSE vs Alpha."),
            ("plot14.png", "KNN: MSE vs. Number of Neighbors")
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

        # Create a frame for the browse section
        browse_frame = tk.Frame(self.frame, bg=BG_COLOR)
        browse_frame.pack(fill="x", pady=(20, 0))
        
        tk.Label(
            browse_frame, 
            text="🔍 Predict Review Scores", 
            font=("Segoe UI", 18, "bold"), 
            bg=BG_COLOR, 
            fg=NAV_BG_COLOR
        ).pack(pady=(0, 10))
        
        tk.Label(
            browse_frame,
            text="Upload a CSV file containing Airbnb listing data to predict review scores.",
            font=("Segoe UI", 12),
            bg=BG_COLOR,
            fg=LABEL_COLOR
        ).pack(pady=(0, 15))

        # Browse button
        browse_btn = tk.Button(
            browse_frame, 
            text="Browse CSV File", 
            font=("Segoe UI", 14, "bold"),
            bg=BUTTON_BG, 
            fg=BUTTON_FG, 
            activebackground=BUTTON_HOVER_BG, 
            activeforeground=BUTTON_FG,
            cursor="hand2", 
            command=self.browse_csv
        )
        browse_btn.pack(pady=(0, 20), ipadx=10, ipady=7)

        # File info label
        self.file_info_label = tk.Label(
            self.frame,
            text="No file selected",
            font=("Segoe UI", 12, "italic"),
            bg=BG_COLOR,
            fg=LABEL_COLOR
        )
        self.file_info_label.pack(pady=(0, 10))

        # Text widget to show CSV contents or predictions
        self.result_text = tk.Text(
            self.frame, 
            width=110, 
            height=22, 
            bg="white", 
            fg=TEXT_COLOR, 
            font=("Consolas", 10),
            bd=2, 
            relief="sunken"
        )
        self.result_text.pack(pady=(0, 20), padx=25)
        self.result_text.config(state="disabled")

        # Predict button below text widget
        self.predict_btn = tk.Button(
            self.frame, 
            text="Run Prediction", 
            font=("Segoe UI", 14, "bold"),
            bg=BUTTON_BG, 
            fg=BUTTON_FG, 
            activebackground=BUTTON_HOVER_BG, 
            activeforeground=BUTTON_FG,
            cursor="hand2", 
            command=self.run_prediction,
            state="disabled"
        )
        self.predict_btn.pack(pady=(0, 30), ipadx=10, ipady=7)

    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            self.csv_data = pd.read_csv(file_path)
            self.csv_path = file_path
            
            # Update file info label
            file_name = os.path.basename(file_path)
            row_count = len(self.csv_data)
            self.file_info_label.config(
                text=f"Selected: {file_name} ({row_count} rows)",
                fg="#27ae60"
            )
            
            # # Display the raw CSV data first
            # self.result_text.config(state="normal")
            # self.result_text.delete("1.0", "end")
            # self.result_text.insert("end", f"CSV Data Preview (first 10 rows):\n\n")
            # self.result_text.insert("end", self.csv_data.head(10).to_string(index=True))
            self.result_text.config(state="disabled")
            
            # Enable the predict button
            self.predict_btn.config(state="normal")

        except Exception as e:
            self.file_info_label.config(
                text=f"Error loading file: {str(e)[:50]}...",
                fg="#e74c3c"
            )
            messagebox.showerror("File Error", f"Failed to load CSV file.\n\nError: {e}")

    def run_prediction(self):
        if self.csv_path is None:
            messagebox.showwarning("No CSV", "Please load a CSV file first.")
            return

        # Show loading message
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "⏳ Running prediction, please wait...\n")
        self.result_text.config(state="disabled")
        self.root.update()

        try:
            # Get the results from prediction function - it returns a tuple
            result = load_and_clean_dataframe(self.csv_path)
            
            # Unpack the tuple - predictions_df is always returned, accuracies may be None
            if isinstance(result, tuple) and len(result) == 3:
                predictions_df, mse_dict,r2_dict = result
            else:
                predictions_df = result
                mse_dict = None
                r2_dict= None

            self.result_text.config(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", "✅ Prediction Results:\n\n")
            self.result_text.insert("end", predictions_df.to_string(index=False))

            self.result_text.config(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", "✅ Prediction Results:\n\n")
            self.result_text.insert("end", predictions_df.to_string(index=False))

            # Display metrics if available
            if mse_dict is not None and r2_dict is not None:
                self.result_text.insert("end", "\n\n📊 Model Performance Metrics:\n")
                for model_name in mse_dict:
                    self.result_text.insert("end", f"\nModel: {model_name}\n")
                    self.result_text.insert("end", f"  MSE: {mse_dict[model_name]:.4f}\n")
                    self.result_text.insert("end", f"  R²: {r2_dict[model_name]:.4f}\n")

            # # Add averages
            # self.result_text.insert("end", "\n\n📈 Prediction Averages:\n")
            # for column in predictions.columns:
            #     avg_value = predictions[column].mean()
            #     self.result_text.insert("end", f"{column}: {avg_value:.2f}\n")
            
            self.result_text.config(state="disabled")

        except Exception as e:
            self.result_text.config(state="normal")
            self.result_text.delete("1.0", "end")
            error_msg = f"❌ Error during prediction:\n\n{str(e)}\n\n{traceback.format_exc()}"
            self.result_text.insert("end", error_msg)
            self.result_text.config(state="disabled")
            messagebox.showerror("Prediction Error", f"Could not perform prediction.\n\nError: {e}")

    def clear_frame(self):
        for widget in self.frame.winfo_children():
            widget.destroy()
        self.images.clear()


if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()