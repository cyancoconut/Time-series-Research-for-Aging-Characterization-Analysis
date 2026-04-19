import pandas as pd


def extract_value(test_name, target):
    # Check if target string is in the test_name
    if any("jri_Aging" in test for test in test_name):
        aging_procedures = [proc for proc in test_name if "Aging" in proc]

        # Look for the part containing the target
        if aging_procedures:
            # take the latest aging procedure in case of multiple
            procedure = aging_procedures[-1]
            if target == "C":
                procedures = procedure.split("_")
                value = procedures[-1]
                return value
            else:
                procedures = procedure.split("_")
                for part in procedures:
                    if target in part:
                        value = part.split(target)[0]
                        return value

    # Return None if no target value is found
    return None


def add_additional_information(df_results):

    df_results["DOD"] = df_results["Procedures"].apply(
        lambda x: extract_value(x, "DOD")
    )
    df_results["DOD"] = df_results["DOD"].apply(lambda x: int(x) if pd.notna(x) else x)

    df_results["SOC"] = df_results["Procedures"].apply(
        lambda x: extract_value(x, "SOC")
    )
    df_results["SOC"] = df_results["SOC"].apply(lambda x: int(x) if pd.notna(x) else x)

    df_results["C_Rate"] = df_results["Procedures"].apply(
        lambda x: extract_value(x, "C")
    )

    df_results["Temperature"] = df_results["Procedures"].apply(
        lambda x: extract_value(x, "grad")
    )
    df_results["Temperature"] = df_results["Temperature"].apply(
        lambda x: int(x) if pd.notna(x) else x
    )
