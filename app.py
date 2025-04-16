import streamlit as st
import pandas as pd
import copy
from pathlib import Path
from datetime import datetime
from simulation import run_simulation
from utils.visualization import Visualization, AdvancedVisualizations
import ast



# Initialize session state
if 'sim_data' not in st.session_state:
    try:
        print("DEBUG: Trying the dataset")
        
        # Directly read the CSV file
        # initial_df = pd.read_csv("data/initial_state_pipeline_filled.csv", encoding="utf-8", errors="replace", converters={'pipeline': eval})
        
        initial_df = pd.read_csv(
            "data/initial_state_pipeline_filled.csv",
            encoding="utf-8",
            # dtype=str,
            on_bad_lines="skip"
        )

                # Convert pipeline column safely
        import ast

        # Convert 'pipeline' column safely (if it exists)
        if 'pipeline' in initial_df.columns:
            def safe_eval(value):
                try:
                    return ast.literal_eval(value) if isinstance(value, str) and value.startswith("[") else []
                except (SyntaxError, ValueError):
                    return []  # If corrupted, return empty list

            initial_df['pipeline'] = initial_df['pipeline'].apply(safe_eval)

        st.session_state.sim_data = initial_df
        st.session_state.last_sim_time = initial_df['time'].max() if 'time' in initial_df.columns else 0

    except Exception as e:
        print(f"DEBUG: Error loading dataset - {e}")
        st.session_state.sim_data = pd.DataFrame()
        st.session_state.last_sim_time = 0

if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {}

def main():
    st.set_page_config(layout="wide")
    st.title("📈 Advanced Inventory Management Digital Twin")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")
        params = get_simulation_parameters()
        
        if st.button("▶️ Run Simulation", key="run_sim_btn"):
            with st.spinner("Running simulation..."):
                initial_state = st.session_state.sim_data if not st.session_state.sim_data.empty else None
                sim = run_simulation(params, initial_state)
                st.session_state.sim_obj = sim
                # Append new simulation data to existing
                # Load simulation results separately, do NOT merge with initial state
                st.session_state.sim_data = pd.DataFrame(sim.data)

        st.header("🔍 Scenario Management")
        scenario_name = st.text_input("Scenario Name", 
                                    f"Scenario_{datetime.now().strftime('%Y%m%d_%H%M')}",
                                    key="scenario_name_input")
        if st.button("💾 Save Scenario", key="save_scenario_btn"):
            st.session_state.scenarios[scenario_name] = {
                'data': st.session_state.sim_data.copy(),
                'params': copy.deepcopy(params)
            }

    # Main display area
    if not st.session_state.sim_data.empty:
        df = st.session_state.sim_data

        Visualization.show_kpi_metrics(df)

        tab1, tab2, tab3, tab4 = st.tabs(["🏭 Operations", "💰 Finance", "📦 Inventory", "📈 Predictions"])

        # 🏭 Operations Tab - Single Column Layout
        with tab1:
            st.subheader("🏭 Operations Analysis")
            st.plotly_chart(Visualization.plot_inventory_dynamics(df), use_container_width=True)
            st.plotly_chart(AdvancedVisualizations.plot_bullwhip_effect(df), use_container_width=True)

            if 'sim_obj' in st.session_state:
                st.plotly_chart(Visualization.plot_lead_time_distribution(st.session_state.sim_obj), use_container_width=True)
                st.plotly_chart(AdvancedVisualizations.plot_order_pipeline(st.session_state.sim_obj), use_container_width=True)

        # 💰 Finance Tab - Single Column Layout
        with tab2:
            st.subheader("💰 Financial Analysis")
            st.plotly_chart(AdvancedVisualizations.plot_cost_tradeoff_3d(df), use_container_width=True)
            st.plotly_chart(Visualization.plot_cost_analysis(df), use_container_width=True)
            st.plotly_chart(AdvancedVisualizations.plot_eoq_heatmap(df), use_container_width=True)

        # 📦 Inventory Tab - Single Column Layout
        with tab3:
            st.subheader("📦 Inventory Status")
            st.plotly_chart(AdvancedVisualizations.plot_inventory_health(df), use_container_width=True)
            st.plotly_chart(Visualization.plot_inventory_tab(df), use_container_width=True)

        # 📈 Predictions Tab - Single Column Layout
        with tab4:
            st.subheader("📈 Demand & Risk Predictions")
            st.plotly_chart(Visualization.plot_forecast(df), use_container_width=True)
            st.plotly_chart(AdvancedVisualizations.plot_stockout_risk(df), use_container_width=True)

    # Scenario comparison
    if st.session_state.scenarios:
        st.subheader("📊 Scenario Benchmarking")
        Visualization.plot_scenario_comparison(st.session_state.scenarios)

def get_simulation_parameters():
    """Collect all required parameters with unique keys"""
    params = {}
    
    # ========== Production Parameters ==========
    st.subheader("🏭 Production Settings")
    prod_col1, prod_col2 = st.columns(2)
    with prod_col1:
        params['production_interval'] = st.slider(
            "Production Interval (days)", 
            0.5, 5.0, 1.0,
            key="production_interval_slider"
        )
    with prod_col2:
        params['production_batch_size'] = st.number_input(
            "Production Batch Size", 
            100, 1000, 500,
            key="production_batch_input"
        )

    # ========== Shipping & Delivery ==========
    st.subheader("🚚 Shipping Settings")
    ship_col1, ship_col2 = st.columns(2)
    with ship_col1:
        params['shipment_delay'] = st.slider(
            "Shipment Delay (days)", 
            1, 7, 2,
            key="shipment_delay_main"
        )
    with ship_col2:
        params['duration'] = st.slider(
            "Simulation Duration (days)", 
            7, 90, 30,
            key="duration_main"
        )
    
    if params['duration'] < params['shipment_delay'] * 2 + 1:
        st.warning(f"⚠️ Minimum recommended duration: {params['shipment_delay'] * 2 + 1} days")
        params['duration'] = max(params['duration'], params['shipment_delay'] * 2 + 1)

    # ========== Inventory Policy ==========
    st.subheader("📦 Inventory Policy")
    policy_col1, policy_col2 = st.columns(2)
    with policy_col1:
        params['policy_type'] = st.selectbox(
            "Policy Type", 
            ['sS', 'ROP'],
            key="policy_type_select"
        )
    with policy_col2:
        params['sS_buffer'] = st.number_input(
            "Buffer Stock (sS only)", 
            50, 500, 200,
            key="ss_buffer_input",
            disabled=(params.get('policy_type', 'sS') != 'sS')
        )
    
    params['safety_stock_factor'] = st.slider(
        "Safety Stock Factor", 
        1.0, 5.0, 2.0,
        key="safety_stock_slider"
    )

    # ========== Cost Parameters ==========
    st.subheader("💰 Cost Settings")
    cost_col1, cost_col2, cost_col3 = st.columns(3)
    with cost_col1:
        params['ordering_cost'] = st.number_input(
            "Ordering Cost ($)", 
            50.0, 500.0, 100.0,
            key="ordering_cost_input"
        )
    with cost_col2:
        params['holding_cost'] = st.number_input(
            "Holding Cost/Unit/Day ($)", 
            0.1, 5.0, 0.5,
            key="holding_cost_input"
        )
    with cost_col3:
        params['shortage_cost'] = st.number_input(
            "Shortage Cost/Unit/Day ($)", 
            0.1, 10.0, 5.0,
            key="shortage_cost_input"
        )

    # ========== Capacity Settings ==========
    st.subheader("🏗️ Capacity Settings")
    cap_col1, cap_col2 = st.columns(2)
    with cap_col1:
        params['initial_warehouse'] = st.number_input(
            "Initial Warehouse Stock", 
            0, 5000, 500,
            key="init_warehouse_input"
        )
        params['warehouse_capacity'] = st.number_input(
            "Warehouse Capacity", 
            500, 10000, 2000,
            key="warehouse_cap_input"
        )
    with cap_col2:
        params['initial_retailer'] = st.number_input(
            "Initial Retailer Stock", 
            0, 1000, 100,
            key="init_retailer_input"
        )
        params['retailer_capacity'] = st.number_input(
            "Retailer Capacity", 
            100, 2000, 500,
            key="retailer_cap_input"
        )

    # ========== Supplier Settings ==========
    st.subheader("🏭 Supplier Settings")
    supp_col1, supp_col2 = st.columns(2)
    with supp_col1:
        params['supplier_reliability'] = st.slider(
            "Supplier Reliability", 
            0.7, 1.0, 0.9,
            key="reliability_slider"
        )
    with supp_col2:
        params['reliability_reduction'] = st.slider(
            "Reliability Reduction During Disruptions", 
            0.1, 0.9, 0.5,
            key="reliability_red_slider"
        )

    # ========== Demand Settings ==========
    st.subheader("📈 Demand Settings")
    demand_col1, demand_col2 = st.columns(2)
    with demand_col1:
        params['average_daily_demand'] = st.number_input(
            "Base Daily Demand", 
            20, 500, 100,
            key="base_demand_input"
        )
        params['linear_growth'] = st.number_input(
            "Daily Demand Growth", 
            0.0, 10.0, 2.0,
            key="demand_growth_input"
        )
    with demand_col2:
        params['seasonal_amplitude'] = st.number_input(
            "Seasonal Amplitude", 
            0.0, 50.0, 15.0,
            key="seasonal_amp_input"
        )
        params['random_std'] = st.number_input(
            "Demand Variability (σ)", 
            0.0, 20.0, 5.0,
            key="demand_var_input"
        )

    # ========== Disruption Settings ==========
    st.subheader("⚡ Disruption Settings")
    disp_col1, disp_col2 = st.columns(2)
    with disp_col1:
        params['disruption_interval'] = st.slider(
            "Disruption Frequency (days)", 
            30, 365, 100,
            key="disruption_freq_slider"
        )
        params['disruption_probability'] = st.slider(
            "Disruption Probability", 
            0.1, 1.0, 0.3,
            key="disruption_prob_slider"
        )
    with disp_col2:
        params['min_downtime'] = st.number_input(
            "Minimum Downtime (days)", 
            1, 7, 2,
            key="min_downtime_input"
        )
        params['max_downtime'] = st.number_input(
            "Maximum Downtime (days)", 
            3, 14, 5,
            key="max_downtime_input"
        )

    # Final validation
    if params['max_downtime'] <= params['min_downtime']:
        st.error("❌ Maximum downtime must be greater than minimum downtime")
        st.stop()
        
    params['demand_multiplier'] = st.selectbox(
        "Demand Scenario", 
        [("Pessimistic", 0.7), ("Neutral", 1.0), ("Optimistic", 1.3)],
        format_func=lambda x: x[0],
        key="demand_scenario_select"
    )[1]

    return params

if __name__ == "__main__":
    main()