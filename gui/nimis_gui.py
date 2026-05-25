import re
import sys
from pathlib import Path
import numpy as np
import torch
import streamlit as st
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from model.nimis_model import NMRPeakSlotAttn

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "nimis_weights.pt"
MAX_COMPONENTS = 4
J_TOL = 0.1
DEVICE = "cpu"

def parse_nmr_string(nmr_text):
    pattern = r"([\d.]+)(?:\s*[-–]\s*[\d.]+)?\s*\((.*?)\)"

    matches = re.findall(pattern, nmr_text)

    peak_positions = []
    integrals = []
    j_values = []
    j_mask = []

    for ppm_str, annotation in matches:
        ppm = float(ppm_str)

        int_match = re.search(r"(\d+(?:\.\d+)?)\s*H", annotation)
        integral = float(int_match.group(1)) if int_match else 1.0

        j_match = re.search(r"J\s*=\s*([\d.,\s]+)\s*Hz", annotation)
        js = []

        if j_match:
            js = [
                float(x.strip())
                for x in re.split(r"[,\s]+", j_match.group(1))
                if x.strip()
            ]

        js = js[:3]
        mask = [1] * len(js)

        while len(js) < 3:
            js.append(0.0)
            mask.append(0)

        peak_positions.append(ppm)
        integrals.append(integral)
        j_values.append(js)
        j_mask.append(mask)

    return {
        "peak_positions_ppm": peak_positions,
        "peak_integrals": integrals,
        "j_values": j_values,
        "j_mask": j_mask }

def parse_manual_peak_table(peak_table):
    peak_positions = []
    integrals = []
    j_values = []
    j_mask = []
    skipped_rows = []

    if isinstance(peak_table, list):
        peak_table = pd.DataFrame(peak_table)

    for idx, row in peak_table.iterrows():
        ppm = row.get("position_ppm")
        integral = row.get("integral")

        if (
            ppm is None
            or integral is None
            or pd.isna(ppm)
            or pd.isna(integral)
            or str(ppm).strip() == ""
            or str(integral).strip() == ""
        ):
            skipped_rows.append(idx + 1)
            continue

        try:
            ppm = float(ppm)
            integral = float(integral)
        except (TypeError, ValueError):
            skipped_rows.append(idx + 1)
            continue

        if not np.isfinite(ppm) or not np.isfinite(integral):
            skipped_rows.append(idx + 1)
            continue

        js = []
        mask = []
        for col in ["J1_Hz", "J2_Hz", "J3_Hz"]:
            value = row.get(col)
            if value is None or pd.isna(value) or str(value).strip() == "":
                js.append(0.0)
                mask.append(0)
            else:
                try:
                    value = float(value)
                    if np.isfinite(value):
                        js.append(value)
                        mask.append(1)
                    else:
                        js.append(0.0)
                        mask.append(0)
                except (TypeError, ValueError):
                    js.append(0.0)
                    mask.append(0)

        peak_positions.append(ppm)
        integrals.append(integral)
        j_values.append(js)
        j_mask.append(mask)

    parsed = {
        "peak_positions_ppm": peak_positions,
        "peak_integrals": integrals,
        "j_values": j_values,
        "j_mask": j_mask}

    return parsed, skipped_rows

def parsed_to_display_table(parsed):
    rows = []
    for ppm, integral, js, mask in zip(
        parsed["peak_positions_ppm"],
        parsed["peak_integrals"],
        parsed["j_values"],
        parsed["j_mask"],
    ):
        row = {
            "position_ppm": ppm,
            "integral": integral,
            "J1_Hz": js[0] if mask[0] else np.nan,
            "J2_Hz": js[1] if mask[1] else np.nan,
            "J3_Hz": js[2] if mask[2] else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows, dtype=float)

def build_con_matrix(integrals):
    integrals = np.asarray(integrals, dtype=np.float32)
    eps = 1e-8

    ratio = integrals[:, None] / (integrals[None, :] + eps)
    ratio = np.where(ratio > 1.0, 1.0 / ratio, ratio)

    return ratio.astype(np.float32)


def build_conj_matrix(j_values, j_mask, tol=0.1):
    j_values = np.asarray(j_values, dtype=np.float32)
    j_mask = np.asarray(j_mask, dtype=bool)

    P = j_values.shape[0]
    conj = np.zeros((P, P), dtype=np.float32)

    for i in range(P):
        js_i = j_values[i][j_mask[i]]
        for j in range(P):
            js_j = j_values[j][j_mask[j]]

            if len(js_i) == 0 or len(js_j) == 0:
                continue

            diff = np.abs(js_i[:, None] - js_j[None, :])

            if np.any(diff <= tol):
                conj[i, j] = 1.0
    return conj


def prepare_inputs(parsed, n_components):
    peak_positions = np.asarray(parsed["peak_positions_ppm"], dtype=np.float32)
    peak_integrals = np.asarray(parsed["peak_integrals"], dtype=np.float32)
    j_values = np.asarray(parsed["j_values"], dtype=np.float32)
    j_mask = np.asarray(parsed["j_mask"], dtype=bool)

    order = np.argsort(peak_positions)
    peak_positions = peak_positions[order]
    peak_integrals = peak_integrals[order]
    j_values = j_values[order]
    j_mask = j_mask[order]

    con = build_con_matrix(peak_integrals)
    conj = build_conj_matrix(j_values, j_mask, tol=J_TOL)

    component_mask = torch.zeros(1, MAX_COMPONENTS, dtype=torch.bool)
    component_mask[:, :n_components] = True

    batch = {
        "peaks": torch.tensor(peak_positions).float().unsqueeze(0),
        "integrals": torch.tensor(peak_integrals).float().unsqueeze(0),
        "con": torch.tensor(con).float().unsqueeze(0),
        "conj": torch.tensor(conj).float().unsqueeze(0),
        "mask": torch.ones(1, len(peak_positions), dtype=torch.bool),
        "component_mask": component_mask,
        "peak_positions": peak_positions,
        "peak_integrals": peak_integrals}
    return batch


def load_model():
    model = NMRPeakSlotAttn(
        dim=128,
        mlp_dim=256,
        depth=8,
        heads=8,
        dim_head=64,
        K=4,
        slot_iters=5,
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def run_prediction(model, batch):
    torch.manual_seed(0)

    with torch.no_grad():
        slots, slot_attn, peak_features, attn_maps = model(
            batch["peaks"],
            batch["integrals"],
            batch["con"],
            mask=batch["mask"],
            component_mask=batch["component_mask"],
            conj=batch["conj"])

    attn = slot_attn.squeeze(0).cpu().numpy()  # (K, P)

    pred_slot = attn.argmax(axis=0)
    confidence = attn.max(axis=0)

    return pred_slot, confidence, attn
st.title("NIMIS: 1H-NMR Mixture Demixing")

input_mode = st.radio(
    "Choose input method",
    ["Paste SI-style NMR string", "Enter peaks manually"],
    horizontal=True)

n_components = st.selectbox("Choose number of mixture components from the drop-down menu", [2, 3, 4], index=2)

if input_mode == "Paste SI-style NMR string":
    nmr_text = st.text_area(
        "Paste SI-style 1H-NMR string",
        height=180,
        value="δ peak position (multiplicity, J = value Hz, value Hz, values Hz, nH), peak position (multiplicity, J = value Hz, value Hz, values Hz, nH), peak position (multiplicity, J = value Hz, value Hz, values Hz, nH)")
    if st.button("Parse NMR string"):
        parsed_preview = parse_nmr_string(nmr_text)
        st.session_state["parsed_si_peaks"] = parsed_preview

    if "parsed_si_peaks" in st.session_state:
        st.subheader("Please check parsed peaks and edit if needed")
        parsed_table = parsed_to_display_table(st.session_state["parsed_si_peaks"])
        confirmed_si_table = st.data_editor(
            parsed_table,
            column_config={
                "position_ppm": st.column_config.NumberColumn("Position (ppm)", format="%.2f"),
                "integral": st.column_config.NumberColumn("Integral", format="%.2f"),
                "J1_Hz": st.column_config.NumberColumn("J1 (Hz)", format="%.2f"),
                "J2_Hz": st.column_config.NumberColumn("J2 (Hz)", format="%.2f"),
                "J3_Hz": st.column_config.NumberColumn("J3 (Hz)", format="%.2f")},
            key="confirmed_si_peak_editor",
            num_rows="dynamic",
            use_container_width=True)
    else:
        confirmed_si_table = None
    manual_table = None
else:
    st.write("Enter one row per peak. Peak position and integral are required. J-coupling fields are optional.")
    manual_table = st.data_editor(
        pd.DataFrame([
            {"position_ppm": 7.26, "integral": 2.0, "J1_Hz": 8.1, "J2_Hz": None, "J3_Hz": None},
            {"position_ppm": 7.12, "integral": 1.0, "J1_Hz": 7.5, "J2_Hz": None, "J3_Hz": None},
            {"position_ppm": 2.31, "integral": 3.0, "J1_Hz": 3.0, "J2_Hz": None, "J3_Hz": None}], dtype=float),
        column_config={
            "position_ppm": st.column_config.NumberColumn("Position (ppm)", format="%.2f"),
            "integral": st.column_config.NumberColumn("Integral", format="%.2f"),
            "J1_Hz": st.column_config.NumberColumn("J1 (Hz)", format="%.2f"),
            "J2_Hz": st.column_config.NumberColumn("J2 (Hz)", format="%.2f"),
            "J3_Hz": st.column_config.NumberColumn("J3 (Hz)", format="%.2f"),
        },
        key="manual_peak_editor",
        num_rows="dynamic",
        use_container_width=True)
    nmr_text = None

if input_mode == "Paste SI-style NMR string":
    run_button = st.button("Confirm and run NIMIS", disabled=("parsed_si_peaks" not in st.session_state))
else:
    run_button = st.button("Run NIMIS")

if run_button:
    if input_mode == "Paste SI-style NMR string":
        parsed, skipped_rows = parse_manual_peak_table(confirmed_si_table)
        if skipped_rows:
            st.warning(
                "Skipped rows with missing peak position/integral: "
                + ", ".join(map(str, skipped_rows))
            )
    else:
        parsed, skipped_rows = parse_manual_peak_table(manual_table)
        if skipped_rows:
            st.warning(
                "Skipped rows with missing peak position/integral: "
                + ", ".join(map(str, skipped_rows)))

    if len(parsed["peak_positions_ppm"]) == 0:
        st.error("No peaks were parsed. Please check the NMR string format.")
    else:
        batch = prepare_inputs(parsed, n_components)
        model = load_model()

        pred_slot, confidence, attn = run_prediction(model, batch)

        result = pd.DataFrame({
            "peak_position_ppm": batch["peak_positions"],
            "integral": batch["peak_integrals"],
            "predicted_component": pred_slot + 1,
            "assignment_confidence": confidence,
        })

        st.subheader("Demixed peak assignments")
        for component_id in sorted(result["predicted_component"].unique()):
            component_df = result.loc[
                result["predicted_component"] == component_id,
                ["peak_position_ppm", "integral", "assignment_confidence"],
            ].reset_index(drop=True)

            st.markdown(f"**Component {int(component_id)}**")
            st.dataframe(component_df, use_container_width=True)