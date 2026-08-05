import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os


class ConfigurationGUI:
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Create main window
        self.root = ctk.CTk()
        self.root.title("Configuration Manager")
        self.root.geometry("800x800")

        # Create main container with padding
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Configuration Settings",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Create input fields
        self.create_input_fields()

        # Create buttons
        self.create_buttons()

        # Initialize values
        self.config_data = {}

    def create_input_fields(self):
        # Project Configuration Section
        self.section_label1 = ctk.CTkLabel(
            self.main_frame,
            text="Project Configuration",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.section_label1.grid(
            row=1, column=0, columnspan=2, pady=(10, 5), sticky="w"
        )

        # Project
        self.project_label = ctk.CTkLabel(self.main_frame, text="Project:")
        self.project_label.grid(row=2, column=0, sticky="w", padx=(20, 10), pady=5)
        self.project_entry = ctk.CTkEntry(self.main_frame, width=400)
        self.project_entry.grid(row=2, column=1, pady=5, padx=(0, 20))

        # Target Specimen
        self.specimen_label = ctk.CTkLabel(self.main_frame, text="Target Specimen:")
        self.specimen_label.grid(row=3, column=0, sticky="w", padx=(20, 10), pady=5)
        self.specimen_entry = ctk.CTkEntry(
            self.main_frame, width=400, placeholder_text="Enter comma-separated values"
        )
        self.specimen_entry.grid(row=3, column=1, pady=5, padx=(0, 20))

        # Test Format
        self.format_label = ctk.CTkLabel(self.main_frame, text="Test Format:")
        self.format_label.grid(row=4, column=0, sticky="w", padx=(20, 10), pady=5)
        self.format_entry = ctk.CTkEntry(
            self.main_frame, width=400, placeholder_text=" Format01"
        )
        self.format_entry.grid(row=4, column=1, pady=5, padx=(0, 20))
        self.format_entry.insert(0, " Format01")

        # Test Name (substring filter on the test name; comma-separated for
        # multiple substrings, matches any; blank falls back to "TS")
        self.name_filter_label = ctk.CTkLabel(self.main_frame, text="Test Name:")
        self.name_filter_label.grid(
            row=5, column=0, sticky="w", padx=(20, 10), pady=5
        )
        self.name_filter_entry = ctk.CTkEntry(
            self.main_frame, width=400, placeholder_text="e.g. TS or TS,EIS"
        )
        self.name_filter_entry.grid(row=5, column=1, pady=5, padx=(0, 20))
        self.name_filter_entry.insert(0, "TS")

        # AHJO Configuration Section
        self.section_label2 = ctk.CTkLabel(
            self.main_frame,
            text="AHJO Configuration",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.section_label2.grid(
            row=6, column=0, columnspan=2, pady=(20, 5), sticky="w"
        )

        # AHJO Endpoint
        self.ahjo_endpoint_label = ctk.CTkLabel(self.main_frame, text="AHJO Endpoint:")
        self.ahjo_endpoint_label.grid(
            row=7, column=0, sticky="w", padx=(20, 10), pady=5
        )
        self.ahjo_endpoint_entry = ctk.CTkEntry(
            self.main_frame,
            width=400,
            placeholder_text="https://ahjo.isea.rwth-aachen.de",
        )
        self.ahjo_endpoint_entry.grid(row=7, column=1, pady=5, padx=(0, 20))
        self.ahjo_endpoint_entry.insert(0, "https://ahjo.isea.rwth-aachen.de")

        # AHJO Key
        self.ahjo_key_label = ctk.CTkLabel(self.main_frame, text="AHJO Key:")
        self.ahjo_key_label.grid(row=8, column=0, sticky="w", padx=(20, 10), pady=5)
        self.ahjo_key_entry = ctk.CTkEntry(self.main_frame, width=400, show="*")
        self.ahjo_key_entry.grid(row=8, column=1, pady=5, padx=(0, 20))

        # MinIO Configuration Section
        self.section_label3 = ctk.CTkLabel(
            self.main_frame,
            text="MinIO Configuration",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.section_label3.grid(
            row=9, column=0, columnspan=2, pady=(20, 5), sticky="w"
        )

        # MinIO Endpoint
        self.minio_endpoint_label = ctk.CTkLabel(
            self.main_frame, text="MinIO Endpoint:"
        )
        self.minio_endpoint_label.grid(
            row=10, column=0, sticky="w", padx=(20, 10), pady=5
        )
        self.minio_endpoint_entry = ctk.CTkEntry(
            self.main_frame,
            width=400,
            placeholder_text="optimusprime.isea.rwth-aachen.de:9000",
        )
        self.minio_endpoint_entry.grid(row=10, column=1, pady=5, padx=(0, 20))
        self.minio_endpoint_entry.insert(0, "optimusprime.isea.rwth-aachen.de:9000")

        # Access Key
        self.access_key_label = ctk.CTkLabel(self.main_frame, text="Access Key:")
        self.access_key_label.grid(row=11, column=0, sticky="w", padx=(20, 10), pady=5)
        self.access_key_entry = ctk.CTkEntry(self.main_frame, width=400)
        self.access_key_entry.grid(row=11, column=1, pady=5, padx=(0, 20))

        # Secret Key
        self.secret_key_label = ctk.CTkLabel(self.main_frame, text="Secret Key:")
        self.secret_key_label.grid(row=12, column=0, sticky="w", padx=(20, 10), pady=5)
        self.secret_key_entry = ctk.CTkEntry(self.main_frame, width=400, show="*")
        self.secret_key_entry.grid(row=12, column=1, pady=5, padx=(0, 20))

        # Bucket Name
        self.bucket_name_label = ctk.CTkLabel(self.main_frame, text="Bucket Name:")
        self.bucket_name_label.grid(row=13, column=0, sticky="w", padx=(20, 10), pady=5)
        self.bucket_name_entry = ctk.CTkEntry(self.main_frame, width=400)
        self.bucket_name_entry.grid(row=13, column=1, pady=5, padx=(0, 20))

        # MinIO Prefix
        self.minio_prefix_label = ctk.CTkLabel(self.main_frame, text="MinIO Prefix:")
        self.minio_prefix_label.grid(
            row=14, column=0, sticky="w", padx=(20, 10), pady=5
        )
        self.minio_prefix_entry = ctk.CTkEntry(
            self.main_frame,
            width=400,
            placeholder_text="j8005-metabatt/Metabatt/VTC",
        )
        self.minio_prefix_entry.grid(row=14, column=1, pady=5, padx=(0, 20))

        # Export Configuration Section
        self.section_label4 = ctk.CTkLabel(
            self.main_frame,
            text="Export Configuration",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.section_label4.grid(
            row=15, column=0, columnspan=2, pady=(20, 5), sticky="w"
        )

        # Export Type
        self.export_type_label = ctk.CTkLabel(self.main_frame, text="Export Type:")
        self.export_type_label.grid(row=16, column=0, sticky="w", padx=(20, 10), pady=5)
        self.export_type_entry = ctk.CTkEntry(
            self.main_frame, width=400, placeholder_text="local"
        )
        self.export_type_entry.grid(row=16, column=1, pady=5, padx=(0, 20))
        self.export_type_entry.insert(0, "local")

        # Export Path
        self.export_path_label = ctk.CTkLabel(self.main_frame, text="Export Path:")
        self.export_path_label.grid(row=17, column=0, sticky="w", padx=(20, 10), pady=5)

        self.path_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.path_frame.grid(row=17, column=1, pady=5, padx=(0, 20), sticky="w")

        self.export_path_entry = ctk.CTkEntry(self.path_frame, width=320)
        self.export_path_entry.pack(side="left", padx=(0, 10))

        self.browse_button = ctk.CTkButton(
            self.path_frame, text="Browse", width=70, command=self.browse_folder
        )
        self.browse_button.pack(side="left")

    def create_buttons(self):
        # Button frame
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.grid(row=18, column=0, columnspan=2, pady=(30, 0))

        # Save button
        self.save_button = ctk.CTkButton(
            self.button_frame,
            text="Save Configuration",
            width=150,
            command=self.save_config,
        )
        self.save_button.pack(side="left", padx=5)

        # Load button
        self.load_button = ctk.CTkButton(
            self.button_frame,
            text="Load Configuration",
            width=150,
            command=self.load_config,
        )
        self.load_button.pack(side="left", padx=5)

        # Apply button
        self.apply_button = ctk.CTkButton(
            self.button_frame,
            text="Apply Values",
            width=150,
            fg_color="green",
            hover_color="darkgreen",
            command=self.apply_values,
        )
        self.apply_button.pack(side="left", padx=5)

        # Clear button
        self.clear_button = ctk.CTkButton(
            self.button_frame,
            text="Clear All",
            width=150,
            fg_color="red",
            hover_color="darkred",
            command=self.clear_all,
        )
        self.clear_button.pack(side="left", padx=5)

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.export_path_entry.delete(0, tk.END)
            self.export_path_entry.insert(0, folder_path)

    def get_config_data(self):
        # Parse target_specimen as list
        specimen_text = self.specimen_entry.get().strip()
        specimen_list = (
            [s.strip() for s in specimen_text.split(",")] if specimen_text else [""]
        )

        # Test-name filter: comma-separated substrings -> list (matches any);
        # blank falls back to "TS" (preserves the historical default).
        name_filter_text = self.name_filter_entry.get().strip()
        name_filter = [
            s.strip() for s in name_filter_text.split(",") if s.strip()
        ] or "TS"

        return {
            "project": self.project_entry.get(),
            "target_specimen": specimen_list,
            "testformat": self.format_entry.get(),
            "name_filter": name_filter,
            "ahjo_endpoint": self.ahjo_endpoint_entry.get(),
            "ahjo_key": self.ahjo_key_entry.get(),
            "minio_endpoint": self.minio_endpoint_entry.get(),
            "access_key": self.access_key_entry.get(),
            "secret_key": self.secret_key_entry.get(),
            "bucket_name": self.bucket_name_entry.get(),
            "minio_prefix": self.minio_prefix_entry.get(),
            "export_type": self.export_type_entry.get(),
            "export_path": self.export_path_entry.get(),
        }

    def save_config(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if file_path:
            config_data = self.get_config_data()
            with open(file_path, "w") as f:
                json.dump(config_data, f, indent=4)
            messagebox.showinfo("Success", "Configuration saved successfully!")

    def load_config(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, "r") as f:
                    config_data = json.load(f)

                # Set values
                self.project_entry.delete(0, tk.END)
                self.project_entry.insert(0, config_data.get("project", ""))

                # Handle target_specimen list
                specimen_list = config_data.get("target_specimen", [""])
                self.specimen_entry.delete(0, tk.END)
                self.specimen_entry.insert(0, ", ".join(specimen_list))

                self.format_entry.delete(0, tk.END)
                self.format_entry.insert(0, config_data.get("testformat", ""))

                nf = config_data.get("name_filter", "TS")
                self.name_filter_entry.delete(0, tk.END)
                self.name_filter_entry.insert(
                    0, ", ".join(nf) if isinstance(nf, list) else str(nf)
                )

                self.ahjo_endpoint_entry.delete(0, tk.END)
                self.ahjo_endpoint_entry.insert(0, config_data.get("ahjo_endpoint", ""))

                self.ahjo_key_entry.delete(0, tk.END)
                self.ahjo_key_entry.insert(0, config_data.get("ahjo_key", ""))

                self.minio_endpoint_entry.delete(0, tk.END)
                self.minio_endpoint_entry.insert(
                    0, config_data.get("minio_endpoint", "")
                )

                self.access_key_entry.delete(0, tk.END)
                self.access_key_entry.insert(0, config_data.get("access_key", ""))

                self.secret_key_entry.delete(0, tk.END)
                self.secret_key_entry.insert(0, config_data.get("secret_key", ""))

                self.bucket_name_entry.delete(0, tk.END)
                self.bucket_name_entry.insert(0, config_data.get("bucket_name", ""))

                self.minio_prefix_entry.delete(0, tk.END)
                self.minio_prefix_entry.insert(0, config_data.get("minio_prefix", ""))

                self.export_type_entry.delete(0, tk.END)
                self.export_type_entry.insert(0, config_data.get("export_type", ""))

                self.export_path_entry.delete(0, tk.END)
                self.export_path_entry.insert(0, config_data.get("export_path", ""))

                messagebox.showinfo("Success", "Configuration loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")

    def clear_all(self):
        # Clear all entry fields
        self.project_entry.delete(0, tk.END)
        self.specimen_entry.delete(0, tk.END)
        self.format_entry.delete(0, tk.END)
        self.name_filter_entry.delete(0, tk.END)
        self.ahjo_endpoint_entry.delete(0, tk.END)
        self.ahjo_key_entry.delete(0, tk.END)
        self.minio_endpoint_entry.delete(0, tk.END)
        self.access_key_entry.delete(0, tk.END)
        self.secret_key_entry.delete(0, tk.END)
        self.bucket_name_entry.delete(0, tk.END)
        self.minio_prefix_entry.delete(0, tk.END)
        self.export_type_entry.delete(0, tk.END)
        self.export_path_entry.delete(0, tk.END)

    def apply_values(self):
        """Apply the configuration values and close GUI"""
        self.config_data = self.get_config_data()
        self.applied = True

        messagebox.showinfo(
            "Success",
            "Values applied successfully! Variables are now available for use.",
        )
        # Close the GUI
        self.root.destroy()

    def get_values(self):
        """Return the configuration values as a dictionary for external use"""
        return self.get_config_data()

    def run(self):
        self.root.mainloop()

        # Return the configuration data after GUI closes
        if self.applied and self.config_data:
            return self.config_data
        else:
            return None


def load_config_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            config_data = json.load(f)
        if config_data:
            # Return individual variables
            return (
                config_data["project"],
                config_data["target_specimen"],
                config_data["testformat"],
                config_data["ahjo_endpoint"],
                config_data["ahjo_key"],
                config_data["minio_endpoint"],
                config_data["access_key"],
                config_data["secret_key"],
                config_data["bucket_name"],
                config_data["minio_prefix"],
                config_data["export_type"],
                config_data["export_path"],
                config_data.get("name_filter", "TS"),
            )
        else:
            # Return defaults if GUI was closed without applying
            return ("", [""], " ", "", "", "", "", "", "", "", "", "", "TS")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
        return None


def get_config_from_gui():
    """Launch GUI and return configuration values"""
    app = ConfigurationGUI()
    config_data = app.run()

    if config_data:
        # Return individual variables
        return (
            config_data["project"],
            config_data["target_specimen"],
            config_data["testformat"],
            config_data["ahjo_endpoint"],
            config_data["ahjo_key"],
            config_data["minio_endpoint"],
            config_data["access_key"],
            config_data["secret_key"],
            config_data["bucket_name"],
            config_data["minio_prefix"],
            config_data["export_type"],
            config_data["export_path"],
            config_data.get("name_filter", "TS"),
        )
    else:
        # Return defaults if GUI was closed without applying
        return ("", [""], " ", "", "", "", "", "", "", "", "", "", "TS")
