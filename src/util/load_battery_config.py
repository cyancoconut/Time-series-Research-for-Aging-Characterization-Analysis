import ipywidgets as widgets
from IPython.display import display, clear_output
import json
from pathlib import Path
import glob


class BatteryConfigJupyter:
    def __init__(self, config_filename="battery_config.json"):
        # Default configuration filename
        self.config_filename = config_filename

        # Create all widgets
        self.create_widgets()

        # Create UI layout
        self.create_layout()

        # Load saved configuration
        self.load_config()

    @property
    def config_file(self):
        """Configuration file path based on working_path and filename from widget"""
        filename = self.config_filename_widget.value.strip()
        if not filename:
            filename = "battery_config.json"
        if not filename.endswith(".json"):
            filename += ".json"
        return Path(self.working_path.value) / filename

    def create_widgets(self):
        """Create all input widgets"""
        # Configuration file name
        self.config_filename_widget = widgets.Text(
            value=self.config_filename,
            description="Config File:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="400px"),
            placeholder="battery_config.json",
        )

        # File and Cell Configuration
        self.working_path = widgets.Text(
            value=r"D:\Data\METABatt",
            description="Working Path:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="600px"),
        )

        self.type_cell = widgets.Text(
            value="A123",
            description="Cell Type:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.nom_capacity = widgets.FloatText(
            value=3.8,
            description="Nominal Capacity (Ah):",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        # Voltage Parameters
        self.v_min = widgets.FloatText(
            value=2.5,
            description="V_min (V):",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.v_max = widgets.FloatText(
            value=3.65,
            description="V_max (V):",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.v_nom = widgets.FloatText(
            value=3.2,
            description="V_nom (V):",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.qocv_crate = widgets.FloatText(
            value=1 / 20,
            description="qOCV C-Rate:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        # Test Parameters
        self.cap_type = widgets.Dropdown(
            options=["CC", "CV", "CCCV"],
            value="CC",
            description="CAP Type:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.cap_rate = widgets.FloatText(
            value=1 / 2,
            description="CAP Rate:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.cap_temp = widgets.IntText(
            value=25,
            description="CAP Temperature (°C):",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.target_pulse_duration = widgets.IntText(
            value=20,
            description="Pulse Duration (s):",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.pulse_type = widgets.IntText(
            value=1,
            description="Pulse Type:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.pulse_target_unit = widgets.Dropdown(
            options=["Resistance", "Voltage", "Current"],
            value="Resistance",
            description="Pulse Target Unit:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        # Processing Parameters
        self.min_rows = widgets.IntText(
            value=20,
            description="Min Rows:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.pau_duration = widgets.FloatText(
            value=9.9,
            description="Pause Duration (min):",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.feature_columns = widgets.Text(
            value="Voltage,Current",
            description="Feature Columns:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="400px"),
        )

        # HDBSCAN Layer 1
        self.hdbscan1_min_cluster = widgets.IntText(
            value=10,
            description="Min Cluster Size:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.hdbscan1_min_samples = widgets.IntText(
            value=10,
            description="Min Samples:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.hdbscan1_epsilon = widgets.FloatText(
            value=3.0,
            description="Epsilon:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.hdbscan1_allow_single = widgets.Checkbox(
            value=False,
            description="Allow Single Cluster",
            style={"description_width": "150px"},
        )

        # HDBSCAN Layer 2
        self.hdbscan2_min_cluster = widgets.IntText(
            value=5,
            description="Min Cluster Size:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.hdbscan2_min_samples = widgets.IntText(
            value=5,
            description="Min Samples:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.hdbscan2_epsilon = widgets.FloatText(
            value=0.08,
            description="Epsilon:",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="300px"),
        )

        self.hdbscan2_allow_single = widgets.Checkbox(
            value=False,
            description="Allow Single Cluster",
            style={"description_width": "150px"},
        )

        # Buttons
        self.save_button = widgets.Button(
            description="Save Config", button_style="success", icon="save"
        )
        self.save_button.on_click(self.save_config)

        self.load_button = widgets.Button(
            description="Load Config", button_style="info", icon="upload"
        )
        self.load_button.on_click(self.load_config)

        self.browse_config_button = widgets.Button(
            description="Browse Config", button_style="", icon="folder-open"
        )
        self.browse_config_button.on_click(self.browse_config)

        self.reset_button = widgets.Button(
            description="Reset Defaults", button_style="warning", icon="refresh"
        )
        self.reset_button.on_click(self.reset_defaults)

        self.generate_button = widgets.Button(
            description="Generate Variables", button_style="primary", icon="play"
        )
        self.generate_button.on_click(self.generate_variables)

        self.export_json_button = widgets.Button(
            description="Export JSON", button_style="", icon="download"
        )
        self.export_json_button.on_click(self.export_json)

        self.show_code_button = widgets.Button(
            description="Show Code", button_style="", icon="code"
        )
        self.show_code_button.on_click(self.show_code)

        # Output area
        self.output = widgets.Output()

    def create_layout(self):
        """Create the UI layout"""
        # General settings
        general_box = widgets.VBox(
            [
                widgets.HTML("<h3>General Settings</h3>"),
                self.config_filename_widget,
                self.working_path,
                self.type_cell,
                self.nom_capacity,
                self.feature_columns,
                self.min_rows,
                self.pau_duration,
            ]
        )

        # Voltage parameters
        voltage_box = widgets.VBox(
            [
                widgets.HTML("<h3>Voltage Parameters</h3>"),
                self.v_min,
                self.v_max,
                self.v_nom,
                self.qocv_crate,
            ]
        )

        # Test parameters
        test_box = widgets.VBox(
            [
                widgets.HTML("<h3>Test Parameters</h3>"),
                self.cap_type,
                self.cap_rate,
                self.cap_temp,
                self.target_pulse_duration,
                self.pulse_type,
                self.pulse_target_unit,
            ]
        )

        # HDBSCAN parameters
        hdbscan_box = widgets.VBox(
            [
                widgets.HTML("<h3>HDBSCAN Parameters</h3>"),
                widgets.HTML("<h4>Layer 1</h4>"),
                self.hdbscan1_min_cluster,
                self.hdbscan1_min_samples,
                self.hdbscan1_epsilon,
                self.hdbscan1_allow_single,
                widgets.HTML("<h4>Layer 2</h4>"),
                self.hdbscan2_min_cluster,
                self.hdbscan2_min_samples,
                self.hdbscan2_epsilon,
                self.hdbscan2_allow_single,
            ]
        )

        # Create tabs
        self.tabs = widgets.Tab()
        self.tabs.children = [general_box, voltage_box, test_box, hdbscan_box]
        self.tabs.set_title(0, "General")
        self.tabs.set_title(1, "Voltage")
        self.tabs.set_title(2, "Test")
        self.tabs.set_title(3, "HDBSCAN")

        self.list_configs_button = widgets.Button(
            description="List Configs", button_style="", icon="list"
        )
        self.list_configs_button.on_click(self.list_available_configs)

        # Buttons - split into two rows for better layout
        buttons_row1 = widgets.HBox(
            [
                self.save_button,
                self.load_button,
                self.browse_config_button,
                self.list_configs_button,
            ]
        )

        buttons_row2 = widgets.HBox(
            [
                self.reset_button,
                self.generate_button,
                self.show_code_button,
                self.export_json_button,
            ]
        )

        buttons_box = widgets.VBox([buttons_row1, buttons_row2])

        # Main layout
        self.main_layout = widgets.VBox(
            [
                widgets.HTML("<h2>Battery Test Configuration</h2>"),
                self.tabs,
                buttons_box,
                self.output,
            ]
        )

    def display(self):
        """Display the UI"""
        display(self.main_layout)

    def get_config_dict(self):
        """Get configuration as dictionary"""
        return {
            "working_path": self.working_path.value,
            "type_cell": self.type_cell.value,
            "nom_capacity": self.nom_capacity.value,
            "v_min": self.v_min.value,
            "v_max": self.v_max.value,
            "v_nom": self.v_nom.value,
            "qocv_crate": self.qocv_crate.value,
            "cap_type": self.cap_type.value,
            "cap_rate": self.cap_rate.value,
            "cap_temp": self.cap_temp.value,
            "target_pulse_duration": self.target_pulse_duration.value,
            "pulse_type": self.pulse_type.value,
            "pulse_target_unit": self.pulse_target_unit.value,
            "min_rows": self.min_rows.value,
            "pau_duration": self.pau_duration.value,
            "feature_columns": [
                col.strip() for col in self.feature_columns.value.split(",")
            ],
            "hdbscan_para_layer_1": {
                "min_cluster_size": self.hdbscan1_min_cluster.value,
                "min_samples": self.hdbscan1_min_samples.value,
                "cluster_selection_epsilon": self.hdbscan1_epsilon.value,
                "allow_single_cluster": self.hdbscan1_allow_single.value,
            },
            "hdbscan_para_layer_2": {
                "min_cluster_size": self.hdbscan2_min_cluster.value,
                "min_samples": self.hdbscan2_min_samples.value,
                "cluster_selection_epsilon": self.hdbscan2_epsilon.value,
                "allow_single_cluster": self.hdbscan2_allow_single.value,
            },
        }

    def set_config_from_dict(self, config):
        """Set configuration from dictionary"""
        self.working_path.value = config.get("working_path", r"D:\Data\METABatt")
        self.type_cell.value = config.get("type_cell", "A123")
        self.nom_capacity.value = config.get("nom_capacity", 3.8)
        self.v_min.value = config.get("v_min", 2.5)
        self.v_max.value = config.get("v_max", 3.65)
        self.v_nom.value = config.get("v_nom", 3.2)
        self.qocv_crate.value = config.get("qocv_crate", 0.05)
        self.cap_type.value = config.get("cap_type", "CC")
        self.cap_rate.value = config.get("cap_rate", 0.5)
        self.cap_temp.value = config.get("cap_temp", 25)
        self.target_pulse_duration.value = config.get("target_pulse_duration", 20)
        self.pulse_type.value = config.get("pulse_type", 1)
        self.pulse_target_unit.value = config.get("pulse_target_unit", "Resistance")
        self.min_rows.value = config.get("min_rows", 20)
        self.pau_duration.value = config.get("pau_duration", 9.9)

        feature_cols = config.get("feature_columns", ["Voltage", "Current"])
        self.feature_columns.value = ",".join(feature_cols)

        # HDBSCAN Layer 1
        hdbscan1 = config.get("hdbscan_para_layer_1", {})
        self.hdbscan1_min_cluster.value = hdbscan1.get("min_cluster_size", 10)
        self.hdbscan1_min_samples.value = hdbscan1.get("min_samples", 10)
        self.hdbscan1_epsilon.value = hdbscan1.get("cluster_selection_epsilon", 3.0)
        self.hdbscan1_allow_single.value = hdbscan1.get("allow_single_cluster", False)

        # HDBSCAN Layer 2
        hdbscan2 = config.get("hdbscan_para_layer_2", {})
        self.hdbscan2_min_cluster.value = hdbscan2.get("min_cluster_size", 5)
        self.hdbscan2_min_samples.value = hdbscan2.get("min_samples", 5)
        self.hdbscan2_epsilon.value = hdbscan2.get("cluster_selection_epsilon", 0.08)
        self.hdbscan2_allow_single.value = hdbscan2.get("allow_single_cluster", False)

    def save_config(self, b=None):
        """Save configuration"""
        with self.output:
            clear_output()
            try:
                # Ensure the working path exists
                working_dir = Path(self.working_path.value)
                working_dir.mkdir(parents=True, exist_ok=True)

                config = self.get_config_dict()
                with open(self.config_file, "w") as f:
                    json.dump(config, f, indent=4)
                print(f"✅ Configuration saved to: {self.config_file}")
            except Exception as e:
                print(f"❌ Error saving configuration: {str(e)}")

    def load_config(self, b=None):
        """Load configuration"""
        with self.output:
            clear_output()
            try:
                if self.config_file.exists():
                    with open(self.config_file, "r") as f:
                        config = json.load(f)
                    self.set_config_from_dict(config)
                    print(f"✅ Configuration loaded from: {self.config_file}")
                else:
                    # Try to load from old location (home directory) for backward compatibility
                    old_config_file = Path.home() / ".battery_config_jupyter.json"
                    if old_config_file.exists():
                        with open(old_config_file, "r") as f:
                            config = json.load(f)
                        self.set_config_from_dict(config)
                        print(
                            f"✅ Configuration loaded from old location: {old_config_file}"
                        )
                        print("ℹ️ Consider saving to update to new location.")
                    else:
                        print(f"ℹ️ No saved configuration found at: {self.config_file}")
            except Exception as e:
                print(f"❌ Error loading configuration: {str(e)}")

    def reset_defaults(self, b=None):
        """Reset to default values"""
        with self.output:
            clear_output()
            default_config = {
                "working_path": r"D:\Data\METABatt",
                "type_cell": "A123",
                "nom_capacity": 3.8,
                "v_min": 2.5,
                "v_max": 3.65,
                "v_nom": 3.2,
                "qocv_crate": 0.05,
                "cap_type": "CC",
                "cap_rate": 0.5,
                "cap_temp": 25,
                "target_pulse_duration": 20,
                "pulse_type": 1,
                "pulse_target_unit": "Resistance",
                "min_rows": 20,
                "pau_duration": 9.9,
                "feature_columns": ["Voltage", "Current"],
                "hdbscan_para_layer_1": {
                    "min_cluster_size": 10,
                    "min_samples": 10,
                    "cluster_selection_epsilon": 3.0,
                    "allow_single_cluster": False,
                },
                "hdbscan_para_layer_2": {
                    "min_cluster_size": 5,
                    "min_samples": 5,
                    "cluster_selection_epsilon": 0.08,
                    "allow_single_cluster": False,
                },
            }
            self.set_config_from_dict(default_config)
            print("✅ Configuration reset to defaults!")

    def generate_variables(self, b=None):
        """Generate variables directly in the notebook namespace"""
        with self.output:
            clear_output()
            config = self.get_config_dict()

            # Get the global namespace
            import IPython

            ip = IPython.get_ipython()

            # Generate all variables
            variables = {
                "working_path": config["working_path"],
                "rootpath": config["working_path"] + r"\BRONZE",
                "type_cell": config["type_cell"],
                "Nom_Capacity": config["nom_capacity"],
                "V_min": config["v_min"],
                "V_max": config["v_max"],
                "V_nom": config["v_nom"],
                "qOCV_CRate": config["qocv_crate"],
                "CAP_Type": config["cap_type"],
                "CAP_Rate": config["cap_rate"],
                "CAP_Temp": config["cap_temp"],
                "target_pulse_duration": config["target_pulse_duration"],
                "pulse_type": config["pulse_type"],
                "pulse_target_unit": config["pulse_target_unit"],
                "MIN_ROWS": config["min_rows"],
                "PAU_DURATION": config["pau_duration"],
                "feature_columns": config["feature_columns"],
                "hdbscan_para_layer_1": config["hdbscan_para_layer_1"],
                "hdbscan_para_layer_2": config["hdbscan_para_layer_2"],
            }

            # Also generate List_Cell
            try:
                rootpath = Path(variables["rootpath"])
                if rootpath.exists():
                    List_Cell = list(rootpath.glob("*.parquet"))
                    List_Cell = [f.name for f in List_Cell]
                    variables["List_Cell"] = List_Cell
                else:
                    print(f"⚠️ Export directory does not exist: {rootpath}")
                    variables["List_Cell"] = []
            except Exception as e:
                print(f"⚠️ Could not get parquet files: {e}")
                variables["List_Cell"] = []

            # Push variables to notebook namespace
            ip.push(variables)

            print("✅ Variables generated successfully!")
            print("\n📋 Available variables:")
            print("=" * 50)
            for var_name, var_value in variables.items():
                if isinstance(var_value, dict):
                    print(f"{var_name} = {json.dumps(var_value, indent=2)}")
                elif isinstance(var_value, list) and var_name == "List_Cell":
                    print(
                        f"{var_name} = {var_value[:3]}..."
                        if len(var_value) > 3
                        else f"{var_name} = {var_value}"
                    )
                else:
                    print(f"{var_name} = {repr(var_value)}")

    def browse_config(self, b=None):
        """Browse and select a config file to load"""
        with self.output:
            clear_output()
            try:
                # Create file selection widget
                from tkinter import filedialog
                import tkinter as tk

                # Create hidden root window
                root = tk.Tk()
                root.withdraw()

                # Open file dialog
                filename = filedialog.askopenfilename(
                    title="Select Configuration File",
                    initialdir=self.working_path.value,
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                )

                root.destroy()

                if filename:
                    # Load the selected config file
                    with open(filename, "r") as f:
                        config = json.load(f)
                    self.set_config_from_dict(config)

                    # Update the config filename widget to reflect the loaded file
                    self.config_filename_widget.value = Path(filename).name

                    print(f"✅ Configuration loaded from: {filename}")
                else:
                    print("ℹ️ No file selected.")
            except ImportError:
                # Fallback for environments where tkinter is not available
                print("⚠️ File browser not available in this environment.")
                print("Please enter the full path to your config file:")

                # Create a temporary text input for file path
                file_input = widgets.Text(
                    placeholder="/path/to/your/config.json",
                    description="Config path:",
                    style={"description_width": "100px"},
                    layout=widgets.Layout(width="500px"),
                )

                load_btn = widgets.Button(description="Load", button_style="success")

                def load_from_path(b):
                    path = file_input.value.strip()
                    if path and Path(path).exists():
                        try:
                            with open(path, "r") as f:
                                config = json.load(f)
                            self.set_config_from_dict(config)
                            self.config_filename_widget.value = Path(path).name
                            print(f"✅ Configuration loaded from: {path}")
                            file_box.close()
                        except Exception as e:
                            print(f"❌ Error loading file: {e}")
                    else:
                        print("❌ Invalid file path")

                load_btn.on_click(load_from_path)

                file_box = widgets.HBox([file_input, load_btn])
                display(file_box)

            except Exception as e:
                print(f"❌ Error browsing for configuration: {str(e)}")

    def list_available_configs(self, b=None):
        """List all available config files in the working directory"""
        with self.output:
            clear_output()
            try:
                working_dir = Path(self.working_path.value)
                if working_dir.exists():
                    config_files = list(working_dir.glob("*.json"))
                    if config_files:
                        print("📋 Available configuration files:")
                        print("=" * 50)
                        for i, file in enumerate(config_files, 1):
                            print(f"{i}. {file.name}")
                        print("=" * 50)
                        print(f"\nTotal: {len(config_files)} config files found")
                    else:
                        print("ℹ️ No configuration files found in working directory")
                else:
                    print(f"⚠️ Working directory does not exist: {working_dir}")
            except Exception as e:
                print(f"❌ Error listing configurations: {str(e)}")

    def show_code(self, b=None):
        """Show Python code without generating variables"""
        with self.output:
            clear_output()
            config = self.get_config_dict()

            code = f"""import glob

# Battery Test Configuration
working_path = '{config['working_path']}'+ "{config['type_cell']}"
rootpath = working_path + r'\\BRONZE'
List_Cell = glob.glob1(rootpath, '*.parquet')

# Cell Parameters
type_cell = "{config['type_cell']}"
Nom_Capacity = {config['nom_capacity']}
V_min = {config['v_min']}
V_max = {config['v_max']}
V_nom = {config['v_nom']}
qOCV_CRate = {config['qocv_crate']}

# Test Parameters
CAP_Type = "{config['cap_type']}"
CAP_Rate = {config['cap_rate']}
CAP_Temp = {config['cap_temp']}
target_pulse_duration = {config['target_pulse_duration']}  # seconds
pulse_type = {config['pulse_type']}  # 1 = single
pulse_target_unit = "{config['pulse_target_unit']}"

# Processing Parameters
MIN_ROWS = {config['min_rows']}  # Minimum number of rows to process a dataframe
PAU_DURATION = {config['pau_duration']}  # PAU in minutes between procedures

# Feature columns
feature_columns = {config['feature_columns']}

# HDBSCAN Parameters
hdbscan_para_layer_1 = {{
    'min_cluster_size': {config['hdbscan_para_layer_1']['min_cluster_size']},
    'min_samples': {config['hdbscan_para_layer_1']['min_samples']},
    'cluster_selection_epsilon': {config['hdbscan_para_layer_1']['cluster_selection_epsilon']},
    'allow_single_cluster': {config['hdbscan_para_layer_1']['allow_single_cluster']}
}}

hdbscan_para_layer_2 = {{
    'min_cluster_size': {config['hdbscan_para_layer_2']['min_cluster_size']},
    'min_samples': {config['hdbscan_para_layer_2']['min_samples']},
    'cluster_selection_epsilon': {config['hdbscan_para_layer_2']['cluster_selection_epsilon']},
    'allow_single_cluster': {config['hdbscan_para_layer_2']['allow_single_cluster']}
}}
"""

            print("📋 Python Code:")
            print("=" * 50)
            print(code)
            print("=" * 50)
            print("\n✅ Code displayed! You can copy it from above.")

    def export_json(self, b=None):
        """Export configuration as JSON"""
        with self.output:
            clear_output()
            config = self.get_config_dict()
            json_str = json.dumps(config, indent=4)

            print("📋 Configuration JSON:")
            print("=" * 50)
            print(json_str)
            print("=" * 50)
            print("\n✅ JSON exported! You can copy it from above.")

    def get_variables(self):
        """Return configuration as variables (for direct use in notebook)"""
        config = self.get_config_dict()

        # Create a namespace with all variables
        class Config:
            pass

        cfg = Config()

        # Basic parameters
        cfg.working_path = config["working_path"]
        cfg.type_cell = config["type_cell"]
        cfg.working_path = cfg.working_path + cfg.type_cell
        cfg.rootpath = cfg.working_path + r"\BRONZE"
        cfg.Nom_Capacity = config["nom_capacity"]
        cfg.V_min = config["v_min"]
        cfg.V_max = config["v_max"]
        cfg.V_nom = config["v_nom"]
        cfg.qOCV_CRate = config["qocv_crate"]
        cfg.CAP_Type = config["cap_type"]
        cfg.CAP_Rate = config["cap_rate"]
        cfg.CAP_Temp = config["cap_temp"]
        cfg.target_pulse_duration = config["target_pulse_duration"]
        cfg.pulse_type = config["pulse_type"]
        cfg.pulse_target_unit = config["pulse_target_unit"]
        cfg.MIN_ROWS = config["min_rows"]
        cfg.PAU_DURATION = config["pau_duration"]
        cfg.feature_columns = config["feature_columns"]
        cfg.hdbscan_para_layer_1 = config["hdbscan_para_layer_1"]
        cfg.hdbscan_para_layer_2 = config["hdbscan_para_layer_2"]

        return cfg
