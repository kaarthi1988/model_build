import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from PIL import Image, ImageTk
import json
import os
import shutil

# Global variables for use inside the GUI function
df_raw = None
user_data_types = {}
user_inputs = {}

def get_inputs_from_gui():
    global df_raw, user_data_types, user_inputs

    # Create the main window
    root = tk.Tk()
    root.title("ML Model Builder")
    root.geometry("700x700")
    
    # Load and display background image
    bg_image = Image.open("GUI_image_new6.png")  # Update with your image file path
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    bg_label.lower()
    
    # --- Define Helper Functions ---
    def load_file():
        """Loads a CSV or Excel file into df_raw."""
        nonlocal root
        global df_raw
        file_path = filedialog.askopenfilename(
            title="Select a CSV or Excel File",
            initialdir="/",  # Optionally specify an initial directory
            filetypes=[("CSV files", "*.csv"),
                       ("Excel files", "*.xlsx"),
                       ("Excel files", "*.xls")]
        )
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    df_raw = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    df_raw = pd.read_excel(file_path)
                else:
                    messagebox.showerror("Invalid File", "Only CSV and Excel files are supported!")
                    return
                lbl_file.config(text=f"Loaded: {file_path.split('/')[-1]}")
                messagebox.showinfo("Success", "File Loaded Successfully!")
                print(df_raw.head())
                # Enable the Data Types button once a file is loaded
                btn_data_types.config(state="normal")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def open_data_types_window():
        """Opens a window to define column data types."""
        nonlocal root
        global user_data_types, df_raw
        if df_raw is None:
            messagebox.showwarning("No Data", "Please upload a file first!")
            return
        data_type_window = tk.Toplevel(root)
        data_type_window.title("Define Column Data Types")
        data_type_window.geometry("800x400")
        frame = tk.Frame(data_type_window)
        frame.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        tk.Label(scrollable_frame, text="Column Name", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
        tk.Label(scrollable_frame, text="Current Type", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=5, pady=5)
        tk.Label(scrollable_frame, text="New Data Type", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=5, pady=5)
        data_type_options = ["integer dtype", "float dtype", "Categorical(Ordinal)", "Categorical(Nominal)", "string", "object dtype", "Datetime dtype"]
        dropdowns = {}
        for i, col_name in enumerate(df_raw.columns, start=1):
            current_type = str(df_raw[col_name].dtype)
            tk.Label(scrollable_frame, text=col_name).grid(row=i, column=0, padx=5, pady=5)
            tk.Label(scrollable_frame, text=current_type).grid(row=i, column=1, padx=5, pady=5)
            new_type_var = tk.StringVar(value="")
            dropdown = ttk.Combobox(scrollable_frame, textvariable=new_type_var, values=data_type_options, state="readonly")
            dropdown.grid(row=i, column=2, padx=5, pady=5)
            dropdowns[col_name] = new_type_var
        
        def apply_changes():
            nonlocal data_type_window
            global user_data_types, df_raw
            user_data_types = {col: var.get() if var.get() else str(df_raw[col].dtype) for col, var in dropdowns.items()}
            print("Final Data Types:", user_data_types)
            messagebox.showinfo("Success", "Data Types Updated!")
            data_type_window.destroy()
        
        btn_apply = tk.Button(scrollable_frame, text="Apply Changes", command=apply_changes, padx=10, pady=5)
        btn_apply.grid(row=len(df_raw.columns) + 1, column=1, pady=10)
    
    def build_model():
        """Collects user inputs and closes the GUI."""
        nonlocal root
        global df_raw, user_data_types, user_inputs
        
        if df_raw is None:
            messagebox.showwarning("No Data", "Please upload a file first!")
            return
        
        # Validate target column (mandatory)
        target_col = entry_target.get().strip()
        if not target_col:
            messagebox.showerror("Error", "Target column is mandatory!")
            return
        if target_col not in df_raw.columns:
            messagebox.showerror("Error", f"Target column '{target_col}' not found in dataframe!")
            return
        
        # Validate remove features (optional)
        remove_cols = entry_remove.get().strip().split(',')
        remove_cols = [col.strip() for col in remove_cols if col.strip()]
        invalid_remove = [col for col in remove_cols if col not in df_raw.columns]
        if invalid_remove:
            messagebox.showerror("Error", f"Invalid columns to remove: {invalid_remove}")
            return
        
        # Validate ordinal degrees (optional)
        ordinal_degrees = entry_ordinal.get().strip().split(',')
        ordinal_degrees = [col.strip() for col in ordinal_degrees if col.strip()]
        
        # Validate binning columns (optional)
        binning_cols = entry_binning.get().strip().split(',')
        binning_cols = [col.strip() for col in binning_cols if col.strip()]
        invalid_binning = [col for col in binning_cols if col not in df_raw.columns]
        if invalid_binning:
            messagebox.showerror("Error", f"Invalid binning columns: {invalid_binning}")
            return
        
        # Get training size (optional; default is 70)
        train_size = train_size_var.get() or "70"
        
        # Get k-fold (optional; default is 10)
        kfold_size = kfold_var.get() or "10"
        
        # Get machine learning task (mandatory)
        ml_task = ml_task_var.get()
        if not ml_task:
            messagebox.showerror("Error", "Machine Learning Task is mandatory!")
            return
        
        # Compile user inputs
        user_inputs = {
            "Target Column": target_col,
            "Remove Features": remove_cols if remove_cols else "None",
            "Ordinal Degree": ordinal_degrees,
            "Binning Columns": binning_cols if binning_cols else "None",
            "Training Data Size": f"{train_size}%",
            "K-Fold": kfold_size,
            "Machine Learning Task": ml_task,
            "Data Types": user_data_types if user_data_types else "Default (inferred from data)"
        }
        
        print("User Inputs:")
        for key, value in user_inputs.items():
            print(f"{key}: {value}")
        
        inputs_summary = "\n".join([f"{key}: {value}" for key, value in user_inputs.items()])
        messagebox.showinfo("Model Building", f"Model building process started with the following inputs:\n\n{inputs_summary}")
        
        # Close the GUI window after collecting the inputs
        root.destroy()
    
    # --- Build the GUI Layout ---
    # Title label
    lbl_title = tk.Label(root, text="ML MODEL BUILDER", font=("Trebuchet MS", 22, "bold"), fg="black", bg="lightblue", borderwidth=4, relief="ridge")
    lbl_title.pack(pady=10)
    
    # Mandatory fields note
    lbl_mandatory = tk.Label(root, text="Fields marked with * are mandatory", fg="red")
    lbl_mandatory.pack()
    
    # File selection frame
    frame_file = tk.Frame(root)
    frame_file.pack(pady=5)
    lbl_file_title = tk.Label(frame_file, text="Upload File *:", font=("Arial", 10))
    lbl_file_title.pack(side="left")
    btn_upload = tk.Button(frame_file, text="Upload File", command=load_file, padx=10, pady=5)
    btn_upload.pack(side="left", padx=10)
    lbl_file = tk.Label(root, text="No file selected", fg="blue")
    lbl_file.pack(pady=5)
    
    # Define Data Types Button (disabled until file is loaded)
    btn_data_types = tk.Button(root, text="Define Data Types", command=open_data_types_window, padx=10, pady=5, state="disabled")
    btn_data_types.pack(pady=10)
    
    # Target Column input
    tk.Label(root, text="Target Column *:", font=("Arial", 10)).pack(pady=2)
    entry_target = tk.Entry(root, width=40)
    entry_target.pack(pady=5)
    
    # Remove Features input
    tk.Label(root, text="Remove Features (comma-separated):", font=("Arial", 10)).pack(pady=2)
    entry_remove = tk.Entry(root, width=40)
    entry_remove.pack(pady=5)
    
    # Ordinal Degree input
    tk.Label(root, text="Ordinal Degree (if applicable):", font=("Arial", 10)).pack(pady=2)
    entry_ordinal = tk.Entry(root, width=40)
    entry_ordinal.pack(pady=5)
    
    # Binning Columns input
    tk.Label(root, text="Binning Columns (comma-separated):", font=("Arial", 10)).pack(pady=2)
    entry_binning = tk.Entry(root, width=40)
    entry_binning.pack(pady=5)
    
    # Training Data Size input
    tk.Label(root, text="Training Data Size (%):", font=("Arial", 10)).pack(pady=2)
    train_size_var = ttk.Combobox(root, values=[str(x) for x in range(60, 100, 5)], state="readonly", width=10)
    train_size_var.set("70")
    train_size_var.pack(pady=5)
    
    # K-Fold input
    tk.Label(root, text="K-Fold:", font=("Arial", 10)).pack(pady=2)
    kfold_var = ttk.Combobox(root, values=["5", "10", "15", "20"], state="readonly", width=10)
    kfold_var.set("10")
    kfold_var.pack(pady=5)
    
    # Machine Learning Task selection
    frame_ml = tk.Frame(root)
    frame_ml.pack(pady=10)
    tk.Label(frame_ml, text="Machine Learning Task *:", font=("Arial", 10)).pack()
    ml_task_var = tk.StringVar()
    frame_radio = tk.Frame(frame_ml)
    frame_radio.pack()
    tasks = [("Regression", "Regression"), ("Classification", "Classification"), ("Clustering", "Clustering")]
    for text, value in tasks:
        tk.Radiobutton(frame_radio, text=text, variable=ml_task_var, value=value).pack(side="left", padx=10)
    
    # Build Model Button
    btn_build_model = tk.Button(root, text="Build Model", command=build_model, padx=10, pady=10)
    btn_build_model.pack(pady=20)
    
    # Start the GUI main loop (this will block until root.destroy() is called)
    root.mainloop()
    
    # After the GUI closes, return the collected inputs, data types, and the loaded DataFrame
    return user_inputs, user_data_types, df_raw

if __name__ == '__main__':
    inputs, data_types, dataframe = get_inputs_from_gui()
    print("Returned User Inputs:")
    print(inputs)
    print("Returned Data Types:")
    print(data_types)
    print("Returned DataFrame:")
    print(dataframe.head() if dataframe is not None else "No DataFrame loaded")
    
    assets_folder = "metadata"
    os.makedirs(assets_folder, exist_ok=True)
    
    json_file_path_user_ip = os.path.join(assets_folder, "user_inputs.json")
    #save the inputs
    with open(json_file_path_user_ip, 'w') as f:
        json.dump(user_inputs, f)
    
    
    json_file_path_user_dtype = os.path.join(assets_folder, "user_data_types.json")
    with open(json_file_path_user_dtype, 'w') as f:
        json.dump(user_data_types, f)
    
    assets_folder = "data"
    os.makedirs(assets_folder, exist_ok=True)
    
    csv_file_path_raw = os.path.join(assets_folder, "df_raw.csv")
    df_raw.to_csv(csv_file_path_raw, index=False)    

    print(f'Inputs are Availabe at {os.getcwd()}')
    