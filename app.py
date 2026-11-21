import json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from exporter import SimulationExporter
from M1_catalog_db import ProductCatalog
from M2_environment import CustomerDemandEnv
from M3_bandit_engine import (
    EpsilonGreedyBandit,
    ThompsonSamplingBandit,
    UCB1Bandit,
)
from M4_ml_layer import ABTestingBaseline, ElasticityModel, RegretTracker


st.set_page_config(
    page_title="OptiPrice - Dynamic Pricing Engine",
    page_icon=":material/payments:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation State
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"


def go_to_simulator():
    st.session_state.current_page = "simulator"


def go_to_home():
    st.session_state.current_page = "home"


# --- LANDING PAGE ---
if st.session_state.current_page == "home":
    st.markdown(
        "<h1 style='color: #00A8E8; margin-bottom: 0px;'>OptiPrice</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='margin-top: 5px; color: #CCCCCC;'>Autonomous Dynamic Pricing Engine using Multi-Armed Bandits</h3>",
        unsafe_allow_html=True,
    )
    st.write("")

    st.markdown(
        """
    Welcome to **OptiPrice**! 
    
    This platform is an automated pricing simulator that helps businesses find the best price for their products in real time. 
    
    Instead of manually setting fixed discounts or relying on static A/B tests, OptiPrice continuously tests different price options, learns how shoppers respond, and automatically adjusts prices to maximize both sales revenue and profit margins.
    """
    )

    st.write("")
    st.button("Launch Price Simulator", on_click=go_to_simulator, type="primary")

    st.divider()

    st.markdown("### How the Optimization Algorithms Work")
    st.write(
        """
        To find the most profitable price, OptiPrice uses smart decision-making algorithms. 
        Each algorithm balances two main tasks: **exploring** (testing different prices to gather shopper data) and **exploiting** (showing the best-performing price to maximize profit).
        """
    )
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1. Epsilon-Greedy ($\\epsilon$-Greedy)")
        st.markdown("**The Straightforward Approach**")
        st.write(
            """
            This algorithm sticks to what works most of the time, but occasionally tries something new. 
            
            * **How it works:** It picks the highest-profit price found so far for the majority of customer visits (e.g., 85% of the time). For the remaining visits (15%), it picks a random price to keep gathering fresh data in case market demand changes.
            """
        )

    with col2:
        st.markdown("#### 2. Upper Confidence Bound (UCB)")
        st.markdown("**The Systematic Explorer**")
        st.write(
            """
            This algorithm makes choices based on how confident it is about each price point.
            
            * **How it works:** It gives extra priority to prices that haven't been tested enough yet. By giving less-tested prices a fair trial first, it ensures no hidden, highly profitable price point is accidentally missed before settling on the winner.
            """
        )

    with col3:
        st.markdown("#### 3. Thompson Sampling")
        st.markdown("**The Probability Learner**")
        st.write(
            """
            This algorithm uses probability models to continuously update its understanding of customer purchasing behavior.
            
            * **How it works:** It estimates the likelihood of each price being the best one. As more shoppers visit and make purchases, the algorithm rapidly directs traffic toward the winning price while smoothly phasing out unprofitable options.
            """
        )

    st.divider()

    st.markdown("### Key System Features")

    a1, a2 = st.columns(2)

    with a1:
        st.info(
            """
            **Built-in Safety Rules**
            * **Price Volatility Protection:** Limits rapid, consecutive price jumps to protect customer trust and brand reputation.
            * **Inventory Awareness:** Helps prevent sudden stockouts by monitoring available product units.
            """
        )

    with a2:
        st.success(
            """
            **Market Adaptability**
            * **Competitor Intelligence:** Automatically adjusts strategy when rival store prices fluctuate.
            * **Demand Spikes:** Handles sudden surges in website visitors and shopping interest.
            """
        )

# --- PRICE SIMULATOR ---
elif st.session_state.current_page == "simulator":
    st.sidebar.button(
        "Back to Landing Page", on_click=go_to_home, use_container_width=True
    )
    st.sidebar.divider()

    st.set_page_config(
        page_title="OptiPrice",
        page_icon=":material/payments:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
        }
        
        .app-header {
            padding-top: 0.5rem;
            padding-bottom: 1.5rem;
            margin-bottom: 1.8rem;
            border-bottom: 1px solid #CBD5E1;
        }
    .app-header h1 {
            font-size: 3.0rem !important;
            font-weight: 800 !important;
            color: #0284C7 !important;
            letter-spacing: -0.035em;
            margin: 0 0 0.2rem 0;
            line-height: 1.1;
        }

        .app-header p {
            font-size: 1.15rem !important;
            color: #64748B !important;
            margin: 0;
            font-weight: 500;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
            border: 1px solid #334155 !important;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .stat-label {
            color: #94A3B8 !important;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .stat-value {
            color: #FFFFFF !important;
            font-size: 1.55rem;
            font-weight: 700;
            margin-top: 4px;
        }
        
        .xai-card {
            background-color: #0F172A;
            border-left: 4px solid #38BDF8;
            border-radius: 6px;
            padding: 14px 18px;
            margin-top: 12px;
            color: #E2E8F0;
            font-size: 0.95rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # HEADER
    st.markdown(
        """
        <div class="app-header">
            <h1>OptiPrice</h1>
            <p>Dynamic Pricing Engine</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # SIDEBAR CONFIGURATION
    st.sidebar.markdown("### Catalog & Product")
    catalog_mode = st.sidebar.radio("Catalog Mode", ["Preset Catalog", "Custom Product"], index=0)

    if catalog_mode == "Preset Catalog":
        preset_name = st.sidebar.selectbox("Product Profile", ProductCatalog.get_preset_names())
        product_config = ProductCatalog.load_preset(preset_name)
    else:
        st.sidebar.markdown("#### Custom Product Parameters")
        c_name = st.sidebar.text_input("Product Name", "Custom Item")
        c_cost = st.sidebar.number_input("Unit Cost (INR)", min_value=1.0, value=1200.0)
        c_min = st.sidebar.number_input("Min Price (INR)", min_value=1.1, value=1500.0)
        c_max = st.sidebar.number_input("Max Price (INR)", min_value=1.2, value=3500.0)
        c_stock = st.sidebar.number_input("Inventory Allocation", min_value=1, value=200)
        c_elasticity = st.sidebar.slider("Price Elasticity", 0.5, 3.5, 1.8, 0.1)
        c_comp = st.sidebar.number_input("Market Competitor Benchmark (INR)", min_value=1.0, value=2500.0)

        if c_min <= c_cost:
            st.sidebar.error("Minimum price must exceed unit cost.")
            st.stop()
        if c_max <= c_min:
            st.sidebar.error("Maximum price must exceed minimum price.")
            st.stop()

        product_config = ProductCatalog.create_custom_product(
            name=c_name,
            unit_cost=c_cost,
            min_price=c_min,
            max_price=c_max,
            initial_stock=c_stock,
            price_elasticity=c_elasticity,
            competitor_price=c_comp
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Pricing Intelligence")

    algorithm_choice = st.sidebar.selectbox(
        "Optimization Algorithm",
        [
            "Thompson Sampling",
            "UCB",
            "Epsilon-Greedy"
        ]
    )

    # Advanced Market Dynamics
    st.sidebar.markdown("### Market Dynamics & Constraints")
    competitor_mode = st.sidebar.selectbox(
        "Competitor Pricing Strategy",
        ["Static Benchmark", "Aggressive Discounter (-12%)", "Premium Follower (+15%)"]
    )

    market_shock = st.sidebar.checkbox("Simulate Demand Surge / Flash Shock (Mid-Simulation)", value=False)
    max_volatility = st.sidebar.slider("Max Price Volatility Jump Cap (INR)", 200, 3000, 1500, 100)

    sim_rounds = st.sidebar.slider("Simulation Visits (t)", 200, 3000, 1000, 100)
    run_button = st.sidebar.button("Run Dynamic Simulation Engine", use_container_width=True, type="primary")

    # PRODUCT OVERVIEW
    st.markdown("##### Product & Competitor Overview")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Product</div><div class="stat-value">{product_config["name"]}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Unit Cost</div><div class="stat-value">&#8377;{product_config["unit_cost"]:,.0f}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Price Range</div><div class="stat-value">&#8377;{product_config["min_price"]:,.0f} - &#8377;{product_config["max_price"]:,.0f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Competitor Price</div><div class="stat-value">&#8377;{product_config.get("competitor_price", 0):,.0f}</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Stock Units</div><div class="stat-value">{product_config["initial_stock"]}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CANDIDATE ARMS
    num_arms = len(product_config["price_arms"])
    st.dataframe(
        pd.DataFrame([product_config["price_arms"]], columns=[f"Arm {i+1}" for i in range(num_arms)]),
        use_container_width=True
    )

    if run_button:
        env_bandit = CustomerDemandEnv(product_config, seed=42)
        env_ab = CustomerDemandEnv(product_config, seed=42)

        arms = product_config["price_arms"]
        cost = product_config["unit_cost"]
        base_comp = product_config.get("competitor_price", (product_config["min_price"] + product_config["max_price"]) / 2)

        # Initialize Bandit Engine
        if algorithm_choice == "Thompson Sampling":
            bandit = ThompsonSamplingBandit(arms, cost)
        elif algorithm_choice == "UCB1":
            bandit = UCB1Bandit(arms, cost, c_factor=1.0)
        else:
            bandit = EpsilonGreedyBandit(arms, cost, epsilon=0.15)

        ab_baseline = ABTestingBaseline(arms, cost)

        exp_profits = [
            env_bandit._calculate_conversion_prob(p, day_of_week=2) * (p - cost)
            for p in arms
        ]
        opt_reward = max(exp_profits)

        tracker_bandit = RegretTracker(opt_reward)
        tracker_ab = RegretTracker(opt_reward)

        trans_logs = []
        prev_bandit_price = arms[0]

        for t in range(sim_rounds):
            dow = t % 7
            
            # Dynamic Competitor Pricing
            if competitor_mode == "Aggressive Discounter (-12%)":
                comp_p = base_comp * 0.88
            elif competitor_mode == "Premium Follower (+15%)":
                comp_p = base_comp * 1.15
            else:
                comp_p = base_comp

            # Market Shock multiplier (midway demand spike)
            shock_mult = 1.6 if (market_shock and (sim_rounds * 0.4 <= t <= sim_rounds * 0.7)) else 1.0

            # Context generation
            ctx_vec, ctx_dict = env_bandit.generate_context(dow)

            # Bandit Arm Selection with Safety Volatility Guardrail
            b_arm = bandit.select_arm()
            candidate_price = arms[b_arm]
            
            # Apply Volatility Cap
            if abs(candidate_price - prev_bandit_price) > max_volatility:
                b_price = prev_bandit_price + np.sign(candidate_price - prev_bandit_price) * max_volatility
                # Snap to closest arm
                b_arm = int(np.argmin([abs(p - b_price) for p in arms]))
                b_price = arms[b_arm]
            else:
                b_price = candidate_price
            prev_bandit_price = b_price

            b_bought, b_reward, b_prob, _ = env_bandit.simulate_step(b_price, dow, ctx_vec, comp_p, shock_mult)
            if algorithm_choice == "Thompson Sampling":
                bandit.update(b_arm, b_reward, b_bought)
            else:
                bandit.update(b_arm, b_reward)
            
            opt_exp = env_bandit._calculate_conversion_prob(arms[np.argmax(exp_profits)], dow, ctx_vec, comp_p, shock_mult) * (arms[np.argmax(exp_profits)] - cost)
            tracker_bandit.log_step(t, algorithm_choice, b_price, b_reward, opt_exp, ctx_dict, comp_p)

            # A/B Testing Step
            ab_arm = ab_baseline.select_arm()
            ab_price = arms[ab_arm]
            ab_bought, ab_reward, ab_prob, _ = env_ab.simulate_step(ab_price, dow, ctx_vec, comp_p, shock_mult)
            ab_baseline.update(ab_arm, ab_reward, ab_bought)
            tracker_ab.log_step(t, "A/B Testing Baseline", ab_price, ab_reward, opt_exp, ctx_dict, comp_p)

            trans_logs.append({
                "price": b_price,
                "purchased": b_bought,
                "device": ctx_dict["device"],
                "tier": ctx_dict["tier"],
                "is_weekend": ctx_dict["is_weekend"],
                "competitor_price": comp_p
            })

        df_b = tracker_bandit.get_dataframe()
        df_ab = tracker_ab.get_dataframe()

        if "price" not in df_b.columns:
            df_b["price"] = [log["price"] for log in trans_logs]

        # Storing logs
        st.session_state["simulation_logs"] = df_b.to_dict(orient="records")

        st.markdown("---")
        st.markdown("##### Executive Simulation Summary")

        b_profit = df_b["cumulative_reward"].iloc[-1]
        ab_profit = df_ab["cumulative_reward"].iloc[-1]
        diff = b_profit - ab_profit
        pct_gain = (diff / max(ab_profit, 1)) * 100
        
        # Identify discovered best arm
        best_arm_idx = int(np.argmax(bandit.values)) if hasattr(bandit, "values") else 0

        if diff > 0:
            diff_color = "#10B981"
            diff_str = f"+&#8377;{diff:,.2f} (+{pct_gain:.1f}%)"
        elif diff < 0:
            diff_color = "#EF4444"
            diff_str = f"-&#8377;{abs(diff):,.2f} ({pct_gain:.1f}%)"
        else:
            diff_color = "#94A3B8"
            diff_str = f"&#8377;0.00 (0.0%)"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="stat-card"><div class="stat-label">{algorithm_choice} Profit</div><div class="stat-value">&#8377;{b_profit:,.2f}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="stat-card"><div class="stat-label">A/B Testing Baseline Profit</div><div class="stat-value">&#8377;{ab_profit:,.2f}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Net AI Profit Gain</div><div class="stat-value" style="color: {diff_color} !important;">{diff_str}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Discovered Optimal Price</div><div class="stat-value">&#8377;{arms[best_arm_idx]:,.2f}</div></div>', unsafe_allow_html=True)

        # EXPLAINABLE AI REASONING CARD
        optimal_margin = arms[best_arm_idx] - cost
        st.markdown(
            f"""
            <div class="xai-card">
                <b>AI Pricing Insights & Reasoning:</b> 
                <b>{algorithm_choice}</b> outperformed static A/B testing by routing <b>{max(bandit.counts)/sum(bandit.counts)*100:.1f}%</b> of traffic toward the profit-maximizing arm of <b>&#8377;{arms[best_arm_idx]:,.2f}</b> (Net Margin: &#8377;{optimal_margin:,.2f}). 
                While A/B testing wasted visitor traffic testing unprofitable price points, the bandit converged dynamically while adapting to competitor shifts (&#8377;{base_comp:,.0f}).
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Profit Trajectory", "Traffic Allocation", "Contextual Breakdown", "Elasticity ML Engine"]
        )

        with tab1:
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Scatter(x=df_b["step"], y=df_b["cumulative_reward"], name=algorithm_choice, line=dict(color="#00E6FF", width=2.5)))
            fig_rev.add_trace(go.Scatter(x=df_ab["step"], y=df_ab["cumulative_reward"], name="A/B Testing Baseline", line=dict(color="#94A3B8", width=2, dash="dash")))
            fig_rev.update_layout(
                title="Cumulative Net Profit Trajectory (INR)",
                xaxis_title="Visitor Interactions (t)",
                yaxis_title="Total Profit (INR)",
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#64748B"),
                xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
                yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
            )
            st.plotly_chart(fig_rev, use_container_width=True)

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=df_b["step"], y=df_b["cumulative_regret"], name=algorithm_choice, line=dict(color="#00E6FF", width=2.5)))
            fig_reg.add_trace(go.Scatter(x=df_ab["step"], y=df_ab["cumulative_regret"], name="A/B Testing Baseline", line=dict(color="#94A3B8", width=2, dash="dash")))
            fig_reg.update_layout(
                title="Cumulative Opportunity Loss (Regret)",
                xaxis_title="Visitor Interactions (t)",
                yaxis_title="Regret (INR)",
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#64748B"),
                xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
                yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
            )
            st.plotly_chart(fig_reg, use_container_width=True)

        with tab2:
            col_t1, col_t2 = st.columns([3, 2])
            with col_t1:
                alloc_counts = df_b["price"].value_counts().reset_index()
                alloc_counts.columns = ["Arm_Price", "Pull Count"]
                alloc_counts["Arm"] = alloc_counts["Arm_Price"].apply(lambda x: f"₹{x:,.0f}")
                alloc_counts = alloc_counts.sort_values(by="Arm_Price")

                fig_alloc = px.bar(alloc_counts, x="Arm", y="Pull Count", title="Traffic Routed to Price Candidates", color_discrete_sequence=["#38BDF8"])
                fig_alloc.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#64748B"),
                    xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
                    yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
                )
                st.plotly_chart(fig_alloc, use_container_width=True)

            with col_t2:
                st.markdown("###### Arm Allocation & Conversion Summary")
                arm_stats_data = []
                for price in arms:
                    arm_logs = [log for log in trans_logs if log["price"] == price]
                    trials = len(arm_logs)
                    conversions = sum(1 for log in arm_logs if log["purchased"] == 1)
                    conv_rate = (conversions / trials) if trials > 0 else 0.0
                    total_profit = conversions * (price - cost)

                    arm_stats_data.append({
                        "Price": f"₹{price:,.0f}",
                        "Margin": f"₹{price - cost:,.0f}",
                        "Trials": trials,
                        "Conversions": conversions,
                        "Conv. Rate": f"{conv_rate:.2%}",
                        "Total Profit": f"₹{total_profit:,.2f}",
                    })
                st.dataframe(pd.DataFrame(arm_stats_data), use_container_width=True)

        with tab3:
            st.markdown("###### Demographic & Context Sensitivity Insights")
            col_c1, col_c2 = st.columns(2)
            df_trans = pd.DataFrame(trans_logs)

            with col_c1:
                dev_summary = df_trans.groupby(["device", "price"])["purchased"].mean().reset_index()
                dev_summary["price_str"] = dev_summary["price"].apply(lambda x: f"₹{x:,.0f}")
                fig_dev = px.bar(dev_summary, x="price_str", y="purchased", color="device", barmode="group",
                                title="Conversion Rate by Device Type (Mobile vs Desktop)",
                                labels={"purchased": "Conversion Rate", "price_str": "Price"})
                fig_dev.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#64748B"))
                st.plotly_chart(fig_dev, use_container_width=True)

            with col_c2:
                tier_summary = df_trans.groupby(["tier", "price"])["purchased"].mean().reset_index()
                tier_summary["price_str"] = tier_summary["price"].apply(lambda x: f"₹{x:,.0f}")
                fig_tier = px.bar(tier_summary, x="price_str", y="purchased", color="tier", barmode="group",
                                title="Conversion Rate by Location Tier (Metro vs Tier-2/3)",
                                labels={"purchased": "Conversion Rate", "price_str": "Price"})
                fig_tier.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#64748B"))
                st.plotly_chart(fig_tier, use_container_width=True)

        with tab4:
            ml_model = ElasticityModel("random_forest")
            ml_model.train(pd.DataFrame(trans_logs))

            dense_p = np.linspace(product_config["min_price"], product_config["max_price"], 30)
            p_conv_desktop = ml_model.predict_demand_curve(dense_p, is_mobile=0, is_metro=1, is_weekend=0)
            p_conv_mobile = ml_model.predict_demand_curve(dense_p, is_mobile=1, is_metro=0, is_weekend=0)

            fig_ml = go.Figure()
            fig_ml.add_trace(go.Scatter(x=dense_p, y=p_conv_desktop, name="Metro Desktop Visitors", line=dict(color="#0284C7", width=2.5)))
            fig_ml.add_trace(go.Scatter(x=dense_p, y=p_conv_mobile, name="Tier-2/3 Mobile Visitors", line=dict(color="#F59E0B", width=2.5, dash="dot")))
            fig_ml.update_layout(
                title="Estimated Demand Curve P(Purchase | Price) across Segments",
                xaxis_title="Price Point (INR)",
                yaxis_title="Predicted Purchase Probability",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#64748B"),
                xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
                yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
            )
            st.plotly_chart(fig_ml, use_container_width=True)

        # EXPORT SECTION
        st.markdown("---")
        st.markdown("##### Export Simulation Logs & Config")
        simulation_records = st.session_state.get("simulation_logs", [])
        exporter = SimulationExporter(log_history=simulation_records, config={"algorithm": algorithm_choice, "timesteps": sim_rounds, "product": product_config["name"]})
        df_export = exporter.to_dataframe()

        if not df_export.empty:
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("Download Performance Logs (CSV)", data=df_export.to_csv(index=False).encode("utf-8"), file_name="optiprice_enterprise_logs.csv", mime="text/csv")
            with col_dl2:
                st.download_button("Download Experiment Config (JSON)", data=json.dumps({"timestamp": datetime.now().isoformat(), "config": exporter.config}, indent=4).encode("utf-8"), file_name="optiprice_config.json", mime="application/json")