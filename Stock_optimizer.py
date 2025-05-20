import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus

# Load data
df = pd.read_csv("mock_products.csv")

# Sidebar logo
st.sidebar.image("logo.png", width=120)

# --- Εκτίμηση Ανάγκης Budget (μόνο πληροφοριακά) ---
if "Cost_per_unit (€)" in df.columns and "Weekly_demand (units)" in df.columns:
    raw_budget = (df["Cost_per_unit (€)"] * df["Weekly_demand (units)"]).sum()
else:
    raw_budget = 2000  # fallback

st.sidebar.header("🎯 Εκτίμηση Ανάγκης Budget")
coverage_target = st.sidebar.slider("Ποσοστό κάλυψης ζήτησης (%)", 10, 100, 80)
suggested_budget = int(raw_budget * (coverage_target / 100))
st.sidebar.markdown(f"📊 Για κάλυψη {coverage_target}% χρειάζεσαι περίπου: **€{suggested_budget}**")

# --- Πραγματικό Budget για Optimization ---
st.sidebar.header("💰 Budget για Optimization")
budget = st.sidebar.number_input("Διαθέσιμο budget (€)", min_value=100, value=suggested_budget, step=50)

# --- Προηγμένες Επιλογές ---
st.sidebar.header("⚙️ Προηγμένες Επιλογές")
use_weights = st.sidebar.checkbox("Χρήση προτεραιοτήτων προϊόντων (weights)")
force_all_products = st.sidebar.checkbox("Αγόρασε τουλάχιστον 1 τεμάχιο από κάθε προϊόν")
min_products = st.sidebar.number_input("Ελάχιστος αριθμός διαφορετικών προϊόντων", min_value=0, value=0, step=1)
min_units_per_product = st.sidebar.number_input("Ελάχιστη ποσότητα ανά προϊόν (αν επιλεγεί)", min_value=0, value=0, step=1)
min_units_all_products = st.sidebar.number_input("Ελάχιστη ποσότητα για ΟΛΑ τα προϊόντα", min_value=0, value=0, step=1)

# --- Τίτλος ---
st.title("🧠 SmartStock Optimizer")
st.markdown("""
**Βρες τη βέλτιστη παραγγελία ανά προϊόν** ώστε:
- Να αξιοποιήσεις σωστά το διαθέσιμο budget
- Να καλύψεις όσο μεγαλύτερο ποσοστό ζήτησης γίνεται
""")

# Show data
st.subheader("📄 Δεδομένα Προϊόντων")
st.dataframe(df)

# Υπολογισμός weight
if use_weights:
    df["Weight"] = round(1 / df["Cost_per_unit (€)"], 2)
else:
    df["Weight"] = 1.0

# Init model
model = LpProblem("Stock_Optimization", LpMaximize)

# Variables
product_vars = {
    row.Product: LpVariable(f"order_{row.Product}", 0, row["Weekly_demand (units)"])
    for _, row in df.iterrows()
}
y_vars = {
    row.Product: LpVariable(f"select_{row.Product}", cat="Binary")
    for _, row in df.iterrows()
}

# Constraints
model += lpSum([
    product_vars[p] * df[df.Product == p]["Cost_per_unit (€)"].values[0]
    for p in product_vars
]) <= budget

if min_units_per_product > 0:
    for p in product_vars:
        model += product_vars[p] >= min_units_per_product * y_vars[p]

if min_units_all_products > 0:
    for p in product_vars:
        model += product_vars[p] >= min_units_all_products

if force_all_products:
    for p in product_vars:
        model += product_vars[p] >= 1

if min_products > 0:
    model += lpSum([y_vars[p] for p in product_vars]) >= min_products

for p in product_vars:
    max_demand = df[df.Product == p]["Weekly_demand (units)"].values[0]
    model += product_vars[p] <= max_demand
    model += product_vars[p] <= 1e5 * y_vars[p]

# Objective: maximize weighted % coverage
model += lpSum([
    product_vars[p] * (df[df.Product == p]["Weight"].values[0] / df[df.Product == p]["Weekly_demand (units)"].values[0])
    for p in product_vars
])

# Solve
status = model.solve()

# Output
if LpStatus[model.status] != "Optimal":
    st.error("❌ Δεν βρέθηκε λύση με το διαθέσιμο budget ή τους περιορισμούς.")
else:
    results = []
    for p in product_vars:
        ordered_units = int(product_vars[p].varValue)
        unit_cost = df[df.Product == p]["Cost_per_unit (€)"].values[0]
        demand = df[df.Product == p]["Weekly_demand (units)"].values[0]
        weight = df[df.Product == p]["Weight"].values[0]
        results.append({
            "Product": p,
            "Ordered": ordered_units,
            "Cost": round(ordered_units * unit_cost, 2),
            "Demand": demand,
            "Coverage (%)": round(100 * ordered_units / demand, 1) if demand > 0 else 0,
            "Weight": weight
        })

    results_df = pd.DataFrame(results)
    nonzero_df = results_df[results_df["Ordered"] > 0]

    st.subheader("📦 Αποτελέσματα Παραγγελίας")
    st.dataframe(results_df.sort_values(by="Coverage (%)", ascending=False))

    st.markdown(f"**Συνολικό Κόστος:** €{results_df['Cost'].sum():.2f}")
    st.markdown(f"**Μέση Κάλυψη Ζήτησης:** {results_df['Coverage (%)'].mean():.1f}%")

    st.subheader("📊 Γραφήματα Ανάλυσης")

    if not nonzero_df.empty:
        top_coverage = nonzero_df.sort_values(by="Coverage (%)", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_coverage["Product"], top_coverage["Coverage (%)"])
        ax.set_xlabel("Κάλυψη (%)")
        ax.set_title("Top 20 προϊόντα με μεγαλύτερη κάλυψη ζήτησης")
        plt.gca().invert_yaxis()
        st.pyplot(fig)

        top_costs = nonzero_df.sort_values(by="Cost", ascending=False).head(20)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(top_costs["Product"], top_costs["Cost"])
        ax2.set_xlabel("Κόστος (€)")
        ax2.set_title("Top 20 προϊόντα με το μεγαλύτερο κόστος προμήθειας")
        plt.gca().invert_yaxis()
        st.pyplot(fig2)
    else:
        st.info("Δεν υπάρχουν προϊόντα με παραγγελία > 0 για να εμφανιστούν διαγράμματα.")
