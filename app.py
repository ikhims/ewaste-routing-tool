"""
E-Waste Routing Decision Support Tool
=====================================

MSc Big Data Analytics dissertation artefact, Sheffield Hallam University.

Predicts the value-optimal end-of-life route (Refurbish / Repair / Recycle / Dispose)
for a used smartphone or tablet from device specifications alone, with a confidence
score and a SHAP-based explanation of each recommendation.

Run locally:      streamlit run app.py
Deployed at:      Streamlit Community Cloud

Phase 7 of 8. Consumes artefacts produced by notebooks 03-06.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

# =============================================================================
# CONFIGURATION
# =============================================================================

ROUTE_ORDER = ["Refurbish", "Repair", "Recycle", "Dispose"]

# Device age is measured against 2021, not the current year. The training corpus is
# a single 2021 market snapshot, so ages computed against today would place every
# device outside the range the model was fitted on. Stated as a limitation in-app.
REFERENCE_YEAR = 2021

TABLET_THRESHOLD_INCHES = 7.0

# Confidence bands measured on the held-out test set in Phase 6.
CONFIDENCE_BANDS = [
    (0.80, "High", "88.5%", "#1a7f37"),
    (0.70, "Moderate", "75.6%", "#4c7a34"),
    (0.60, "Fair", "69.4%", "#9a6700"),
    (0.50, "Low", "61.1%", "#bc4c00"),
    (0.00, "Very low", "58.8%", "#cf222e"),
]

REVIEW_THRESHOLD = 0.70  # below this, the tool recommends human review

ROUTE_DESCRIPTIONS = {
    "Refurbish": "High residual value with strong proportional retention. Worth full "
                 "refurbishment — testing, cleaning, consumable replacement and warranty — "
                 "for resale into the premium second-hand market.",
    "Repair": "Sufficient residual value to justify repairing a fault. Resale value "
              "exceeds typical UK repair cost with margin remaining.",
    "Recycle": "Below the point where repair or resale recovers its own handling cost. "
               "Materials and component recovery is the indicated route.",
    "Dispose": "No economically indicated repair, resale or component-recovery route. "
               "Route to compliant WEEE disposal.",
}

# Brands present in the training corpus. Anything else encodes as 'Others'.
CORPUS_BRANDS = [
    "Acer", "Alcatel", "Asus", "Gionee", "HTC", "Honor", "Huawei", "LG", "Lava",
    "Lenovo", "Meizu", "Micromax", "Motorola", "Nokia", "Oppo", "Panasonic",
    "Realme", "Samsung", "Sony", "XOLO", "Xiaomi", "ZTE", "Others",
]

# =============================================================================
# NAMED UK REFERENCE DEVICES
# =============================================================================
# Specifications sourced from published manufacturer and GSMArena data (Aug 2026),
# NOT from the training corpus. This is a different data source from the one the
# model was fitted on, and is disclosed as such in the methodology.
#
# rear_camera_mp records the HIGHEST ADVERTISED rear sensor, matching how
# marketplace listings headline camera specification — which is the most likely
# convention behind the single camera figure in the training corpus. For devices
# whose highest sensor is not the main wide lens (notably the Galaxy S21, where
# 64 MP is the telephoto), this is flagged in the device note.

NAMED_DEVICES = {
    "Samsung Galaxy A32 4G (64GB)": dict(
        reference_gbp=99.39,
        brand="Samsung", screen_in=6.4, rear_mp=64, front_mp=20, ram=4,
        storage=64, battery=5000, weight=184, year=2021, has_4g=1, has_5g=0,
        note="A 5G version exists with a different display, camera array and chipset.",
    ),
    "Samsung Galaxy A52 4G (128GB)": dict(
        reference_gbp=123.66,
        brand="Samsung", screen_in=6.5, rear_mp=64, front_mp=32, ram=6,
        storage=128, battery=4500, weight=189, year=2021, has_4g=1, has_5g=0,
        note="RAM shipped at 4/6/8 GB for this storage tier; 6 GB assumed as the common UK retail configuration.",
    ),
    "Samsung Galaxy S21 (128GB)": dict(
        reference_gbp=165.25,
        brand="Samsung", screen_in=6.2, rear_mp=64, front_mp=10, ram=8,
        storage=128, battery=4000, weight=169, year=2021, has_4g=1, has_5g=1,
        note="The 64 MP sensor is the telephoto, not the main wide lens (12 MP). "
             "Recorded as the highest advertised figure to match listing convention.",
    ),
    "Samsung Galaxy S21 (256GB)": dict(
        reference_gbp=190.38,
        brand="Samsung", screen_in=6.2, rear_mp=64, front_mp=10, ram=8,
        storage=256, battery=4000, weight=169, year=2021, has_4g=1, has_5g=1,
        note="Identical to the 128 GB variant in every specification except storage.",
    ),
    "Samsung Galaxy Tab A7 10.4 (32GB)": dict(
        reference_gbp=89.66,
        brand="Samsung", screen_in=10.4, rear_mp=8, front_mp=5, ram=3,
        storage=32, battery=7040, weight=476, year=2020, has_4g=1, has_5g=0,
        note="LTE figures shown; the Wi-Fi-only model has no cellular capability.",
    ),
    "Samsung Galaxy Tab S7 (128GB)": dict(
        reference_gbp=216.05,
        brand="Samsung", screen_in=11.0, rear_mp=13, front_mp=8, ram=6,
        storage=128, battery=8000, weight=498, year=2020, has_4g=1, has_5g=0,
        note="Wi-Fi model weight. Separate LTE and 5G variants exist.",
    ),
    "Xiaomi Redmi Note 10 4G (128GB)": dict(
        reference_gbp=107.67,
        brand="Xiaomi", screen_in=6.43, rear_mp=48, front_mp=13, ram=4,
        storage=128, battery=5000, weight=178.8, year=2021, has_4g=1, has_5g=0,
        note="Distinct from the Note 10S, Note 10 Pro and Note 10 5G, which are different devices.",
    ),
    "Motorola Moto G Power 2021 (64GB)": dict(
        reference_gbp=49.78,
        brand="Motorola", screen_in=6.6, rear_mp=48, front_mp=8, ram=4,
        storage=64, battery=5000, weight=206.5, year=2021, has_4g=1, has_5g=0,
        note="2020 and 2022 model years differ substantially — the 2020 model has a 16 MP main camera.",
    ),
    "Google Pixel 5 (128GB)": dict(
        reference_gbp=168.98,
        brand="Others", screen_in=6.0, rear_mp=12.2, front_mp=8, ram=8,
        storage=128, battery=4080, weight=151, year=2020, has_4g=1, has_5g=1,
        note="OUT-OF-CORPUS BRAND: Google does not appear in the training data and is "
             "encoded as 'Others'. The model has no Google-specific pricing signal.",
    ),
    "Google Pixel 6 (128GB)": dict(
        reference_gbp=130.83,
        brand="Others", screen_in=6.4, rear_mp=50, front_mp=8, ram=8,
        storage=128, battery=4614, weight=207, year=2021, has_4g=1, has_5g=1,
        note="OUT-OF-CORPUS BRAND: Google is encoded as 'Others'. The 50 MP sensor "
             "outputs 12.5 MP binned images.",
    ),
    "OnePlus 9 (128GB)": dict(
        reference_gbp=173.66,
        brand="Others", screen_in=6.55, rear_mp=50, front_mp=16, ram=8,
        storage=128, battery=4500, weight=192, year=2021, has_4g=1, has_5g=1,
        note="OUT-OF-CORPUS BRAND: OnePlus is encoded as 'Others'. The 50 MP sensor "
             "is the ultrawide; the main wide lens is 48 MP.",
    ),
    "OnePlus Nord (128GB)": dict(
        reference_gbp=133.33,
        brand="Others", screen_in=6.44, rear_mp=48, front_mp=32, ram=8,
        storage=128, battery=4115, weight=184, year=2020, has_4g=1, has_5g=1,
        note="OUT-OF-CORPUS BRAND: OnePlus is encoded as 'Others'. Dual front camera (32 MP + 8 MP).",
    ),
}

# Apple devices are listed so evaluators can see the scope boundary, but the model
# never predicts on them — see the scope notice in render_apple_notice().
APPLE_DEVICES = {
    "Apple iPhone 11 (64GB)": dict(year=2019, screen_in=6.1),
    "Apple iPhone 13 (128GB)": dict(year=2021, screen_in=6.1),
    "Apple iPad 9th Gen (64GB)": dict(year=2021, screen_in=10.2),
}

# Plain-language names for SHAP explanations
FEATURE_LABELS = {
    "total_camera_mp": "combined camera resolution",
    "rear_camera_mp": "rear camera resolution",
    "front_camera_mp": "front camera resolution",
    "camera_ratio": "front-to-rear camera balance",
    "screen_size_inches": "screen size",
    "is_tablet": "tablet form factor",
    "internal_memory": "storage capacity",
    "log_internal_memory": "storage capacity",
    "storage_to_ram_ratio": "storage-to-memory balance",
    "ram": "RAM",
    "battery": "battery capacity",
    "weight": "device weight",
    "battery_per_gram": "battery density",
    "4g": "4G capability",
    "5g": "5G capability",
    "connectivity_tier": "connectivity generation",
    "device_age_years": "device age",
    "years_used": "time in previous use",
    "usage_ratio": "proportion of life spent in use",
    "rear_camera_imputed": "multi-camera array indicator",
}


def humanise(feature_name: str) -> str:
    """Convert a feature column name into something a practitioner would recognise."""
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    if feature_name.startswith("brand_grouped_"):
        return f"brand ({feature_name.replace('brand_grouped_', '').title()})"
    if feature_name.startswith("os_"):
        return f"operating system ({feature_name.replace('os_', '').title()})"
    return feature_name.replace("_", " ")


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_artefacts():
    """Load the trained model and build the SHAP explainer once per session."""
    root = Path(__file__).parent
    model_files = sorted((root / "outputs" / "models").glob("final_model_*.joblib"))
    if not model_files:
        st.error("No trained model found in outputs/models/. Run notebook 05 first.")
        st.stop()

    bundle = joblib.load(model_files[-1])
    model = bundle["model"]

    # Probability and SHAP columns follow the model's own class order, which is
    # alphabetical — not the project's value ordering. Reordering is mandatory;
    # a mismatch here silently corrupts both the confidence figure and every
    # explanation, and produced a below-chance ROC-AUC during Phase 5.
    classes = [str(c) for c in model.classes_]
    class_index = [classes.index(r) for r in ROUTE_ORDER]

    return model, bundle["feature_columns"], class_index, shap.TreeExplainer(model)


@st.cache_data
def load_exemplars():
    """Load corpus exemplars spanning all four routes and a range of confidence levels."""
    root = Path(__file__).parent
    path = root / "data" / "processed" / "app_exemplars.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# =============================================================================
# FEATURE CONSTRUCTION
# =============================================================================

def build_feature_row(spec: dict, years_used: float, feature_columns: list) -> pd.DataFrame:
    """
    Assemble a single-row feature matrix from base device specifications.

    Mirrors the Phase 3 feature engineering exactly. Any divergence between this
    function and notebook 03 would produce predictions the model was not trained
    to make, so the derived formulas are kept identical.
    """
    age = max(REFERENCE_YEAR - spec["year"], 1)   # floor at 1 to avoid division by zero
    screen_in = spec["screen_in"]

    row = {
        "screen_size_inches": round(screen_in, 2),
        "is_tablet": int(screen_in >= TABLET_THRESHOLD_INCHES),
        "rear_camera_mp": spec["rear_mp"],
        "front_camera_mp": spec["front_mp"],
        "total_camera_mp": spec["rear_mp"] + spec["front_mp"],
        "camera_ratio": round(spec["front_mp"] / (spec["rear_mp"] + 0.01), 4),
        "internal_memory": spec["storage"],
        "log_internal_memory": round(float(np.log1p(spec["storage"])), 4),
        "ram": spec["ram"],
        "storage_to_ram_ratio": round(spec["storage"] / spec["ram"], 3),
        "battery": spec["battery"],
        "weight": spec["weight"],
        "battery_per_gram": round(spec["battery"] / spec["weight"], 3),
        "4g": spec["has_4g"],
        "5g": spec["has_5g"],
        "connectivity_tier": spec["has_4g"] + spec["has_5g"],
        "device_age_years": age,
        "years_used": round(years_used, 4),
        "usage_ratio": round(years_used / age, 4),
        "rear_camera_imputed": 0,
    }

    # One-hot encodings, all zeroed then set
    for col in feature_columns:
        if col.startswith(("brand_grouped_", "os_")):
            row[col] = 0

    brand_col = f"brand_grouped_{spec['brand'].lower()}"
    if brand_col in feature_columns:
        row[brand_col] = 1
    elif "brand_grouped_others" in feature_columns:
        row["brand_grouped_others"] = 1

    if "os_android" in feature_columns:
        row["os_android"] = 1

    return pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)


# =============================================================================
# PREDICTION AND EXPLANATION
# =============================================================================

def predict(model, explainer, class_index, X: pd.DataFrame):
    """Return the predicted route, full probability vector and SHAP contributions."""
    proba = model.predict_proba(X)[0][class_index]
    route = ROUTE_ORDER[int(np.argmax(proba))]

    shap_raw = np.array(explainer.shap_values(X))
    shap_vals = shap_raw[0, :, class_index[ROUTE_ORDER.index(route)]] \
        if shap_raw.ndim == 3 else shap_raw[0]

    return route, proba, shap_vals


def confidence_band(confidence: float):
    """Map a confidence score onto its Phase 6 measured accuracy band."""
    for threshold, label, accuracy, colour in CONFIDENCE_BANDS:
        if confidence >= threshold:
            return label, accuracy, colour
    return "Very low", "58.8%", "#cf222e"


def top_contributions(shap_vals, X, feature_columns, n=4):
    """Return the n features moving the prediction most, with direction and value."""
    contributions = pd.DataFrame({
        "feature": feature_columns,
        "shap": shap_vals,
        "value": X.iloc[0].to_numpy(),
    })
    contributions["magnitude"] = contributions["shap"].abs()
    return contributions.nlargest(n, "magnitude")


# =============================================================================
# UI COMPONENTS
# =============================================================================

def route_implied_by_price(gbp: float) -> str:
    """
    The route the Phase 4 thresholds would assign given an observed sterling value.

    Refurbish additionally requires above-median proportional retention, which cannot be
    determined from a used price alone, so devices above the Refurbish floor are reported
    as 'Refurbish or Repair' rather than resolved to one.
    """
    if gbp >= 100.0:
        return "Refurbish or Repair"
    if gbp >= 75.0:
        return "Repair"
    if gbp >= 40.0:
        return "Recycle"
    return "Dispose"


def render_price_check(route: str, reference_gbp: float):
    """
    Compare the model's recommendation against the route implied by independently
    collected UK market prices.

    This is the closest thing the project has to an external check on the routing
    label, since the calibration prices were gathered from CeX, eBay UK and Back Market
    rather than derived from the training corpus. It is shown rather than hidden: an
    asset-management professional will know these prices, and a tool that conceals a
    discrepancy they can spot themselves loses more trust than one that surfaces it.
    """
    implied = route_implied_by_price(reference_gbp)
    agrees = route in implied

    st.markdown("#### Check against UK market price")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Observed UK used price", f"£{reference_gbp:,.2f}")
        st.caption("Consensus of CeX, eBay UK and Back Market, Aug 2026")
    with right:
        if agrees:
            st.success(
                f"**Consistent.** At £{reference_gbp:,.2f} the routing thresholds imply "
                f"**{implied}**, which includes the model's recommendation of {route}."
            )
        else:
            st.error(
                f"**Discrepancy.** At £{reference_gbp:,.2f} the routing thresholds imply "
                f"**{implied}**, but the model recommends **{route}** from specifications alone. "
                "The model over-weights specification tier relative to observed market price for "
                "this device — a real limitation, shown rather than hidden."
            )
    st.caption(
        "The model never sees price. This comparison is an independent check, not an input."
    )


def render_apple_notice(device_name: str):
    """Apple devices are outside the model's scope — say so rather than guessing."""
    st.warning(f"**{device_name} falls outside this tool's scope.**")
    st.markdown(
        """
Apple devices were excluded from the training corpus. Two reasons:

- No CC0-licensed Apple pricing source was available for the sterling calibration layer,
  so Apple residual values could not be anchored on the same basis as Android devices.
- Apple devices depreciate on a systematically flatter curve than Android. With only 39
  Apple records in the source data, the model would have learnt an unreliable
  Apple-specific pattern and applied it confidently to unseen devices.

**The model would return a prediction if asked, and that prediction would be wrong in a
predictable direction** — it would under-value the device and route it too aggressively
towards Recycle. Returning a scope message is the more honest behaviour.

Supporting Apple properly would require Apple records back in the training corpus, Apple
reference devices in the calibration layer, and the routing thresholds re-derived. That is
outside the scope of this dissertation and is noted as further work.
        """
    )


def render_recommendation(route: str, proba: np.ndarray, shap_vals, X, feature_columns):
    """Render the routing recommendation, confidence and explanation."""
    confidence = float(proba.max())
    band, band_accuracy, colour = confidence_band(confidence)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Recommended route")
        st.markdown(
            f"<div style='background:{colour}15;border-left:4px solid {colour};"
            f"padding:14px 18px;border-radius:4px;'>"
            f"<div style='font-size:1.9rem;font-weight:600;color:{colour};'>{route}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(ROUTE_DESCRIPTIONS[route])

    with right:
        st.markdown("#### Model confidence")
        st.metric(label=f"{band} confidence", value=f"{confidence:.0%}")
        st.caption(
            f"Predictions in this confidence band were **{band_accuracy} accurate** on the "
            f"held-out test set."
        )

    if confidence < REVIEW_THRESHOLD:
        st.warning(
            f"**Human review recommended.** At {confidence:.0%} confidence this device sits "
            f"below the {REVIEW_THRESHOLD:.0%} threshold where the model's accuracy is "
            "materially lower. Treat the recommendation as a prompt, not a decision."
        )

    st.markdown("#### Probability across all four routes")
    prob_df = pd.DataFrame({"Route": ROUTE_ORDER, "Probability": proba}).set_index("Route")
    st.bar_chart(prob_df, height=200)

    st.markdown("#### Why this recommendation")
    st.caption(
        f"The attributes moving this device most strongly towards or away from **{route}**, "
        "measured by SHAP contribution."
    )

    contributions = top_contributions(shap_vals, X, feature_columns)
    for _, r in contributions.iterrows():
        direction = "towards" if r["shap"] > 0 else "away from"
        arrow = "▲" if r["shap"] > 0 else "▼"
        colour = "#1a7f37" if r["shap"] > 0 else "#cf222e"
        value = f"{r['value']:,.1f}".rstrip("0").rstrip(".")
        st.markdown(
            f"<span style='color:{colour};font-weight:600;'>{arrow}</span> "
            f"**{humanise(r['feature']).capitalize()}** ({value}) pushes "
            f"*{direction}* {route} &nbsp;·&nbsp; "
            f"<span style='color:#666;'>contribution {r['shap']:+.3f}</span>",
            unsafe_allow_html=True,
        )


def render_limitations():
    """Scope and limitation disclosures. Kept visible rather than buried."""
    with st.expander("Scope, limitations and how to read these recommendations"):
        st.markdown(
            """
**What this tool does.** Predicts a value-optimal end-of-life route from device
specifications alone. It never sees price data — that is the point. In practice an
operator holding a returned device has the specification and needs the value band.

**Accuracy.** Macro-F1 of 0.685 and macro ROC-AUC of 0.906 on a held-out test set of 656
devices. In plain terms: the recommendation is exactly right about 69% of the time, and
within one route of correct over 99% of the time. **This is a triage aid, not an automated
router.** A human retains the decision.

**Where it is weakest.** The Refurbish/Repair boundary. Refurbish requires both high
absolute value *and* above-median proportional value retention; the retention condition is
invisible to a model that sees only specifications, so devices near that boundary cannot be
resolved in principle.

**Known limitations, stated plainly:**

- **No condition data.** The training corpus contains no cosmetic or functional grading, so
  the model cannot see whether a device is damaged. Recommendations assume a functional
  device of typical condition.
- **Apple not supported.** See the scope notice when an Apple device is selected.
- **Google and OnePlus are out-of-corpus brands.** Neither appears in the training data;
  both encode as 'Others', so no brand-specific signal is available for them.
- **Label uncertainty.** The routing labels were engineered from a sterling calibration
  layer of 12 reference devices. Re-anchoring within that layer's own source-spread band
  reassigns up to 52.5% of devices between routes. The class boundaries carry real
  uncertainty and the recommendations inherit it.
- **Age is measured against 2021.** The training corpus is a 2021 market snapshot. Ages are
  computed relative to that year to keep inputs within the range the model was fitted on.
- **Marketplace selection bias.** Training devices were listed for sale, so genuinely
  worthless devices are largely absent. The tool is likely to under-predict Dispose on
  exactly the devices where that route is most appropriate.
- **Named device specifications come from published manufacturer data**, a different source
  from the training corpus.
            """
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.set_page_config(page_title="E-Waste Routing Decision Tool",
                       page_icon="♻️", layout="wide")

    st.title("E-Waste Routing Decision Support Tool")
    st.markdown(
        "Predicts the value-optimal end-of-life route for a used smartphone or tablet — "
        "**Refurbish, Repair, Recycle or Dispose** — from device specifications alone."
    )
    st.caption(
        "MSc Big Data Analytics dissertation artefact · Sheffield Hallam University · "
        "Research prototype, not a commercial product"
    )
    st.divider()

    model, feature_columns, class_index, explainer = load_artefacts()
    exemplars = load_exemplars()

    # ---- Sidebar --------------------------------------------------------
    with st.sidebar:
        st.header("Select a device")

        source = st.radio(
            "Device source",
            ["Named UK devices", "Corpus exemplars", "Enter specifications manually"],
            help="Named devices use published manufacturer specifications. Corpus exemplars "
                 "are real records from the dataset the model was trained on.",
        )

        st.divider()
        if source == "Enter specifications manually":
            years_used = st.slider(
                "Time in previous use (years)", 0.0, 3.0, 1.5, 0.25,
                help="How long the previous owner used the device. Distinct from device age.",
                )
        else:
            years_used = 1.5  # placeholder; overwritten from the device row below

    # ---- Device selection ------------------------------------------------
    spec, device_name, device_note = None, None, None

    if source == "Named UK devices":
        options = ["— select a device —"] + list(NAMED_DEVICES) + list(APPLE_DEVICES)
        device_name = st.selectbox("Device", options)

        if device_name in APPLE_DEVICES:
            render_apple_notice(device_name)
            render_limitations()
            return
        if device_name in NAMED_DEVICES:
            spec = NAMED_DEVICES[device_name]
            device_note = spec.get("note")

    elif source == "Corpus exemplars":
        if exemplars is None:
            st.error("Exemplar file not found. Run notebook 06 to regenerate it.")
            return

        labels = ["— select a device —"] + exemplars["display_name"].tolist()
        choice = st.selectbox("Device", labels)

        if choice != "— select a device —":
            row = exemplars.loc[exemplars["display_name"] == choice].iloc[0]
            device_name = choice
            spec = dict(
                brand=row["brand"], screen_in=row["screen_size_inches"],
                rear_mp=row["rear_camera_mp"], front_mp=row["front_camera_mp"],
                ram=row["ram"], storage=row["internal_memory"],
                battery=row["battery"], weight=row["weight"],
                year=int(REFERENCE_YEAR - row["device_age_years"]),
                has_4g=int(row["4g"]), has_5g=int(row["5g"]),
            )
            years_used = float(row["years_used"])

            st.info(
    "**Before revealing the recommendation:** which route would you assign this "
    "device? These exemplars are real records from the training data, each carrying "
    "the route assigned by this study's own labelling rule. That rule is derived from "
    "resale values rather than verified outcomes, so treat it as a reference point "
    "rather than a correct answer — where you disagree with it, that disagreement is "
    "useful data."
)
            if not st.checkbox("Show the model's recommendation"):
                st.markdown("#### Device specifications")
                st.dataframe(pd.DataFrame([{
                    "Brand": spec["brand"], "Screen (in)": spec["screen_in"],
                    "Rear camera (MP)": spec["rear_mp"], "Front camera (MP)": spec["front_mp"],
                    "RAM (GB)": spec["ram"], "Storage (GB)": spec["storage"],
                    "Battery (mAh)": spec["battery"], "Weight (g)": spec["weight"],
                    "Age (years)": row["device_age_years"],
                }]), hide_index=True, use_container_width=True)
                render_limitations()
                return

    else:
        st.markdown("#### Enter device specifications")
        c1, c2, c3 = st.columns(3)
        with c1:
            brand = st.selectbox("Brand", CORPUS_BRANDS, index=CORPUS_BRANDS.index("Samsung"))
            screen_in = st.number_input("Screen size (inches)", 2.0, 13.0, 6.4, 0.1)
            year = st.number_input("Release year", 2013, 2021, 2020, 1)
        with c2:
            rear_mp = st.number_input("Rear camera (MP)", 0.0, 200.0, 48.0, 1.0)
            front_mp = st.number_input("Front camera (MP)", 0.0, 60.0, 16.0, 1.0)
            weight = st.number_input("Weight (g)", 50.0, 900.0, 190.0, 1.0)
        with c3:
            storage = st.selectbox("Storage (GB)", [8, 16, 32, 64, 128, 256, 512], index=4)
            ram = st.selectbox("RAM (GB)", [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0], index=4)
            battery = st.number_input("Battery (mAh)", 500, 12000, 4000, 100)

        c4, c5 = st.columns(2)
        with c4:
            has_4g = int(st.checkbox("4G capable", value=True))
        with c5:
            has_5g = int(st.checkbox("5G capable", value=False))

        spec = dict(brand=brand, screen_in=screen_in, rear_mp=rear_mp, front_mp=front_mp,
                    ram=ram, storage=storage, battery=battery, weight=weight,
                    year=year, has_4g=has_4g, has_5g=has_5g)
        device_name = "Manually specified device"

    if spec is None:
        st.info("Select a device from the sidebar to see a routing recommendation.")
        render_limitations()
        return

    # ---- Prediction ------------------------------------------------------
    X = build_feature_row(spec, years_used, feature_columns)
    route, proba, shap_vals = predict(model, explainer, class_index, X)

    st.divider()
    st.subheader(device_name)
    if device_note:
        st.caption(f"Note: {device_note}")

    render_recommendation(route, proba, shap_vals, X, feature_columns)

    if spec.get("reference_gbp"):
        st.divider()
        render_price_check(route, spec["reference_gbp"])

    with st.expander("Specifications used for this prediction"):
        st.dataframe(pd.DataFrame([{
            "Brand": spec["brand"],
            "Brand encoding": ("Others (out-of-corpus)"
                               if f"brand_grouped_{spec['brand'].lower()}" not in feature_columns
                               else spec["brand"]),
            "Screen (in)": spec["screen_in"], "Rear camera (MP)": spec["rear_mp"],
            "Front camera (MP)": spec["front_mp"], "RAM (GB)": spec["ram"],
            "Storage (GB)": spec["storage"], "Battery (mAh)": spec["battery"],
            "Weight (g)": spec["weight"],
            "Device age (years)": REFERENCE_YEAR - spec["year"],
            "Time in use (years)": years_used,
        }]).T.rename(columns={0: "Value"}), use_container_width=True)

    render_limitations()


if __name__ == "__main__":
    main()
