import streamlit as st
import math
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="가화실 - 가상 화학 실험실",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.card {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    background-color: #ffffff;
    margin-bottom: 15px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.experiment-title {
    font-size: 23px;
    font-weight: 700;
}

.metric-box {
    padding: 15px;
    border-radius: 15px;
    background-color: #f5f7fb;
    text-align: center;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
}

.safe-box {
    padding: 16px;
    border-radius: 14px;
    background-color: #eefbf3;
    border: 1px solid #b7ebc6;
}

.warning-box {
    padding: 16px;
    border-radius: 14px;
    background-color: #fff8e6;
    border: 1px solid #f1d48a;
}

.beaker {
    width: 260px;
    height: 300px;
    border: 6px solid #555;
    border-top: none;
    border-radius: 0 0 35px 35px;
    margin: 20px auto;
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        to bottom,
        rgba(255,255,255,0.4),
        rgba(255,255,255,0.1)
    );
}

.liquid {
    position: absolute;
    bottom: 0;
    width: 100%;
    transition: all 0.5s ease;
}

.precipitate {
    position: absolute;
    bottom: 0;
    width: 100%;
    height: 35px;
    background: rgba(220,220,220,0.9);
}

.reagent-drop {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background-color: #4c8bf5;
    margin: 5px auto;
}

.small-text {
    font-size: 13px;
    color: #666;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 세션 상태
# ============================================================

if "records" not in st.session_state:
    st.session_state.records = []

if "titration_added" not in st.session_state:
    st.session_state.titration_added = 0.0

if "neutral_added" not in st.session_state:
    st.session_state.neutral_added = 0.0

if "precip_added" not in st.session_state:
    st.session_state.precip_added = 0.0

if "experiment_result" not in st.session_state:
    st.session_state.experiment_result = None


# ============================================================
# 안전한 화학 계산 함수
# ============================================================

def safe_log10(x):
    return math.log10(max(x, 1e-14))


def calculate_moles(M, V_ml):
    """몰수 계산"""
    return M * (V_ml / 1000)


def calculate_molarity(moles, V_ml):
    """몰농도 계산"""
    if V_ml <= 0:
        return 0
    return moles / (V_ml / 1000)


def calculate_dilution(M1, V1_ml, V2_ml):
    """M1V1=M2V2"""
    if V2_ml <= 0:
        return 0
    return M1 * V1_ml / V2_ml


def strong_acid_ph(moles_h, total_volume_l):
    """강산의 이상적 pH"""
    if total_volume_l <= 0:
        return 7.0

    concentration = moles_h / total_volume_l

    if concentration <= 1e-14:
        return 7.0

    return max(0.0, min(14.0, -safe_log10(concentration)))


def strong_base_ph(moles_oh, total_volume_l):
    """강염기의 이상적 pH"""
    if total_volume_l <= 0:
        return 7.0

    concentration = moles_oh / total_volume_l

    if concentration <= 1e-14:
        return 7.0

    poh = -safe_log10(concentration)
    return max(0.0, min(14.0, 14 - poh))


def neutralization_ph(
    acid_M,
    acid_ml,
    base_M,
    base_ml
):
    """
    강산-강염기 중화의 단순 이상 모델.
    H+ + OH- -> H2O
    """

    acid_moles = calculate_moles(acid_M, acid_ml)
    base_moles = calculate_moles(base_M, base_ml)

    total_l = (acid_ml + base_ml) / 1000

    if total_l <= 0:
        return 7.0

    remaining = acid_moles - base_moles

    if abs(remaining) < 1e-12:
        return 7.0

    if remaining > 0:
        return strong_acid_ph(remaining, total_l)

    return strong_base_ph(abs(remaining), total_l)


def titration_ph(
    acid_M,
    acid_ml,
    base_M,
    base_ml
):
    """
    강산-강염기 적정 모델.

    equivalence point:
        Ca * Va = Cb * Vb
    """

    acid_moles = acid_M * acid_ml / 1000
    base_moles = base_M * base_ml / 1000

    total_l = (acid_ml + base_ml) / 1000

    if total_l <= 0:
        return 7.0

    difference = acid_moles - base_moles

    if abs(difference) < 1e-12:
        return 7.0

    if difference > 0:
        return strong_acid_ph(difference, total_l)

    return strong_base_ph(abs(difference), total_l)


def precipitation_amount(
    ca_M,
    ca_ml,
    co3_M,
    co3_ml
):
    """
    Ca2+ + CO3(2-) -> CaCO3(s)

    1:1 반응으로 교육용 모델링.
    """

    ca_moles = calculate_moles(ca_M, ca_ml)
    co3_moles = calculate_moles(co3_M, co3_ml)

    precip_moles = min(ca_moles, co3_moles)

    remaining_ca = max(0, ca_moles - precip_moles)
    remaining_co3 = max(0, co3_moles - precip_moles)

    return precip_moles, remaining_ca, remaining_co3


def ph_color(ph):
    """
    pH에 따른 교육용 색상 표현.
    실제 지시약의 정확한 색상 모델이 아닌 시각화용 모델.
    """

    if ph < 3:
        return "#ef4444"
    elif ph < 5:
        return "#f97316"
    elif ph < 6:
        return "#facc15"
    elif ph < 7:
        return "#fde68a"
    elif ph < 8:
        return "#86efac"
    elif ph < 9:
        return "#22c55e"
    elif ph < 11:
        return "#06b6d4"
    else:
        return "#3b82f6"


def save_record(name, experiment, conditions, result):
    record = {
        "시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "실험": experiment,
        "이름": name,
        "조건": conditions,
        "결과": result
    }

    st.session_state.records.append(record)


# ============================================================
# 제목
# ============================================================

st.markdown(
    '<div class="main-title">🧪 가화실</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">가상 화학 실험실 · 안전하게 배우고, 직접 실험하고, 결과를 분석하세요.</div>',
    unsafe_allow_html=True
)


# ============================================================
# 사이드바
# ============================================================

with st.sidebar:

    st.header("🔬 실험실 메뉴")

    menu = st.radio(
        "이동",
        [
            "🏠 실험실 홈",
            "🧪 산-염기 중화",
            "📈 산-염기 적정",
            "🧴 용액 제조·희석",
            "⚗️ 침전 반응",
            "🧮 화학 계산기",
            "📚 실험 기록"
        ]
    )

    st.divider()

    st.markdown("""
    <div class="safe-box">
    <b>🛡️ 안전한 가상 실험</b><br><br>
    이 웹앱은 교육용 가상 시뮬레이션입니다.
    실제 화학물질이나 장비를 사용하지 않습니다.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 홈
# ============================================================

if menu == "🏠 실험실 홈":

    st.header("🏠 가상 화학 실험실")

    st.write(
        "실제 실험실의 비용과 안전상의 제약 없이 "
        "화학 반응과 계산 원리를 가상 환경에서 체험할 수 있습니다."
    )

    st.divider()

    cols = st.columns(2)

    experiments = [
        (
            "🧪",
            "산-염기 중화 실험",
            "산과 염기를 섞으면서 pH와 중화 반응을 관찰합니다."
        ),
        (
            "📈",
            "산-염기 적정 실험",
            "시약을 조금씩 첨가하면서 적정곡선과 당량점을 확인합니다."
        ),
        (
            "🧴",
            "용액 제조·희석",
            "몰농도, 몰수, 희석 전후 농도 변화를 학습합니다."
        ),
        (
            "⚗️",
            "침전 반응",
            "두 용액을 섞어 불용성 물질이 생성되는 과정을 관찰합니다."
        )
    ]

    for i, exp in enumerate(experiments):

        with cols[i % 2]:

            icon, title, description = exp

            st.markdown(
                f"""
                <div class="card">
                    <div style="font-size:40px">{icon}</div>
                    <div class="experiment-title">{title}</div>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.subheader("🎯 학습 목표")

    goal_cols = st.columns(4)

    goals = [
        ("🧮", "계산", "몰수·몰농도·pH"),
        ("👀", "관찰", "색 변화·침전"),
        ("📊", "분석", "그래프·당량점"),
        ("📝", "기록", "실험 결과 저장")
    ]

    for col, (icon, title, desc) in zip(goal_cols, goals):

        with col:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div style="font-size:30px">{icon}</div>
                    <b>{title}</b><br>
                    <span class="small-text">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# 산-염기 중화
# ============================================================

elif menu == "🧪 산-염기 중화":

    st.header("🧪 산-염기 중화 실험")

    st.write(
        "강산과 강염기의 중화 반응을 가상으로 관찰합니다."
    )

    st.info("핵심 원리: H⁺ + OH⁻ → H₂O")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("① 산 용액")

        acid_M = st.number_input(
            "산 농도 (mol/L)",
            min_value=0.001,
            max_value=5.0,
            value=0.100,
            step=0.01,
            key="neutral_acid_M"
        )

        acid_ml = st.number_input(
            "산 부피 (mL)",
            min_value=1.0,
            max_value=500.0,
            value=50.0,
            step=1.0,
            key="neutral_acid_ml"
        )

        st.caption("가상 시약: HCl")

    with col2:

        st.subheader("② 염기 용액")

        base_M = st.number_input(
            "염기 농도 (mol/L)",
            min_value=0.001,
            max_value=5.0,
            value=0.100,
            step=0.01,
            key="neutral_base_M"
        )

        add_ml = st.slider(
            "추가할 염기 부피 (mL)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key="neutral_add_ml"
        )

        st.caption("가상 시약: NaOH")

    current_ph = neutralization_ph(
        acid_M,
        acid_ml,
        base_M,
        add_ml
    )

    acid_moles = calculate_moles(acid_M, acid_ml)
    base_moles = calculate_moles(base_M, add_ml)

    total_volume = acid_ml + add_ml

    st.divider()

    result_cols = st.columns(4)

    values = [
        ("현재 pH", f"{current_ph:.3f}"),
        ("산 몰수", f"{acid_moles:.5f} mol"),
        ("염기 몰수", f"{base_moles:.5f} mol"),
        ("총 부피", f"{total_volume:.1f} mL")
    ]

    for col, (label, value) in zip(result_cols, values):

        with col:

            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="small-text">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.subheader("🫙 실시간 비커")

    liquid_height = min(
        90,
        max(20, total_volume / 150 * 100)
    )

    color = ph_color(current_ph)

    st.markdown(
        f"""
        <div class="beaker">
            <div class="liquid"
                 style="
                 height:{liquid_height}%;
                 background-color:{color};
                 opacity:0.75;">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"<center><b>현재 pH: {current_ph:.2f}</b></center>",
        unsafe_allow_html=True
    )

    # pH 변화 그래프

    additions = np.linspace(0, 100, 101)

    ph_values = [
        neutralization_ph(
            acid_M,
            acid_ml,
            base_M,
            x
        )
        for x in additions
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=additions,
            y=ph_values,
            mode="lines",
            name="pH"
        )
    )

    fig.add_vline(
        x=acid_M * acid_ml / base_M,
        line_dash="dash",
        annotation_text="이론적 당량점"
    )

    fig.update_layout(
        title="염기 첨가량에 따른 pH 변화",
        xaxis_title="NaOH 첨가량 (mL)",
        yaxis_title="pH",
        yaxis=dict(range=[0, 14]),
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    if st.button("💾 중화 실험 결과 저장"):

        save_record(
            "중화 실험",
            "산-염기 중화",
            {
                "산 농도": acid_M,
                "산 부피": acid_ml,
                "염기 농도": base_M,
                "염기 첨가량": add_ml
            },
            {
                "최종 pH": round(current_ph, 4),
                "산 몰수": acid_moles,
                "염기 몰수": base_moles
            }
        )

        st.success("실험 결과가 기록되었습니다.")


# ============================================================
# 적정 실험
# ============================================================

elif menu == "📈 산-염기 적정":

    st.header("📈 산-염기 적정 실험")

    st.write(
        "뷰렛에서 염기 용액을 조금씩 첨가하여 "
        "pH 변화를 관찰하고 당량점을 찾습니다."
    )

    col1, col2 = st.columns(2)

    with col1:

        acid_M = st.number_input(
            "분석 대상 산 농도 (mol/L)",
            min_value=0.001,
            max_value=2.0,
            value=0.100,
            step=0.01,
            key="titration_acid_M"
        )

        acid_ml = st.number_input(
            "산 시료 부피 (mL)",
            min_value=1.0,
            max_value=200.0,
            value=25.0,
            step=1.0,
            key="titration_acid_ml"
        )

    with col2:

        base_M = st.number_input(
            "적정 염기 농도 (mol/L)",
            min_value=0.001,
            max_value=2.0,
            value=0.100,
            step=0.01,
            key="titration_base_M"
        )

        step_ml = st.number_input(
            "한 번에 첨가할 양 (mL)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="titration_step"
        )

    theoretical_eq = acid_M * acid_ml / base_M

    st.info(
        f"이론적 당량점: **{theoretical_eq:.2f} mL**"
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("➕ 염기 첨가"):

            st.session_state.titration_added += step_ml

            if st.session_state.titration_added > 100:
                st.session_state.titration_added = 100

    with c2:

        if st.button("🔄 초기화"):

            st.session_state.titration_added = 0.0
            st.rerun()

    with c3:

        st.metric(
            "현재 첨가량",
            f"{st.session_state.titration_added:.1f} mL"
        )

    current_added = st.session_state.titration_added

    current_ph = titration_ph(
        acid_M,
        acid_ml,
        base_M,
        current_added
    )

    st.subheader("🧪 현재 실험 상태")

    r1, r2, r3 = st.columns(3)

    r1.metric("현재 pH", f"{current_ph:.3f}")
    r2.metric("염기 첨가량", f"{current_added:.1f} mL")

    if theoretical_eq > 0:

        error = current_added - theoretical_eq

        r3.metric(
            "당량점까지",
            f"{error:+.2f} mL"
        )

    st.subheader("🫙 실험 기구")

    color = ph_color(current_ph)

    st.markdown(
        f"""
        <div class="beaker">
            <div class="liquid"
                 style="
                 height:65%;
                 background-color:{color};
                 opacity:0.75;">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 적정곡선

    volumes = np.linspace(
        0.01,
        max(100, theoretical_eq * 2),
        400
    )

    curve = [
        titration_ph(
            acid_M,
            acid_ml,
            base_M,
            v
        )
        for v in volumes
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=volumes,
            y=curve,
            mode="lines",
            name="적정곡선"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[theoretical_eq],
            y=[7],
            mode="markers",
            marker=dict(size=12),
            name="당량점"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[current_added],
            y=[current_ph],
            mode="markers",
            marker=dict(size=14),
            name="현재 위치"
        )
    )

    fig.update_layout(
        title="산-염기 적정곡선",
        xaxis_title="염기 첨가량 (mL)",
        yaxis_title="pH",
        yaxis=dict(range=[0, 14]),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    if abs(current_added - theoretical_eq) < step_ml:

        st.success(
            "🎯 현재 상태가 이론적인 당량점에 매우 가깝습니다."
        )

    if st.button("💾 적정 결과 저장"):

        save_record(
            "적정 실험",
            "산-염기 적정",
            {
                "산 농도": acid_M,
                "산 부피": acid_ml,
                "염기 농도": base_M,
                "현재 첨가량": current_added
            },
            {
                "현재 pH": round(current_ph, 4),
                "이론적 당량점": round(theoretical_eq, 4)
            }
        )

        st.success("적정 결과가 저장되었습니다.")


# ============================================================
# 용액 제조 및 희석
# ============================================================

elif menu == "🧴 용액 제조·희석":

    st.header("🧴 용액 제조 및 희석 실험")

    st.write(
        "몰농도와 몰수의 관계 및 M₁V₁ = M₂V₂를 이용한 "
        "희석 원리를 학습합니다."
    )

    tabs = st.tabs(
        [
            "🧮 몰농도 계산",
            "💧 희석 계산",
            "🧪 가상 제조"
        ]
    )

    with tabs[0]:

        st.subheader("몰농도 계산")

        c1, c2 = st.columns(2)

        with c1:

            moles = st.number_input(
                "용질의 몰수 (mol)",
                min_value=0.0001,
                value=0.1000,
                step=0.01,
                key="moles_calc"
            )

        with c2:

            volume = st.number_input(
                "용액 부피 (mL)",
                min_value=0.1,
                value=500.0,
                step=10.0,
                key="volume_calc"
            )

        M = calculate_molarity(
            moles,
            volume
        )

        st.success(
            f"몰농도 = **{M:.4f} mol/L**"
        )

        st.latex(
            r"M=\frac{n}{V}"
        )

    with tabs[1]:

        st.subheader("희석 계산")

        c1, c2, c3 = st.columns(3)

        with c1:

            M1 = st.number_input(
                "원액 농도 M₁",
                min_value=0.001,
                value=1.0,
                step=0.1
            )

        with c2:

            V1 = st.number_input(
                "사용한 원액 V₁ (mL)",
                min_value=0.1,
                value=10.0,
                step=1.0
            )

        with c3:

            V2 = st.number_input(
                "최종 부피 V₂ (mL)",
                min_value=0.1,
                value=100.0,
                step=5.0
            )

        M2 = calculate_dilution(
            M1,
            V1,
            V2
        )

        st.success(
            f"희석 후 농도 M₂ = **{M2:.4f} mol/L**"
        )

        st.latex(
            r"M_1V_1=M_2V_2"
        )

    with tabs[2]:

        st.subheader("🧪 가상 용액 제조")

        target_M = st.number_input(
            "목표 농도 (mol/L)",
            min_value=0.001,
            value=0.100,
            step=0.01
        )

        target_V = st.number_input(
            "목표 부피 (mL)",
            min_value=1.0,
            value=100.0,
            step=10.0
        )

        molecular_weight = st.number_input(
            "용질 몰질량 (g/mol)",
            min_value=1.0,
            value=58.44,
            step=0.1
        )

        required_moles = (
            target_M * target_V / 1000
        )

        required_mass = (
            required_moles * molecular_weight
        )

        r1, r2 = st.columns(2)

        r1.metric(
            "필요한 몰수",
            f"{required_moles:.5f} mol"
        )

        r2.metric(
            "계산된 용질 질량",
            f"{required_mass:.4f} g"
        )

        st.info(
            "이 화면의 제조 과정은 계산 원리를 시각화한 "
            "가상 시뮬레이션입니다."
        )


# ============================================================
# 침전 반응
# ============================================================

elif menu == "⚗️ 침전 반응":

    st.header("⚗️ 침전 반응 실험")

    st.write(
        "두 수용액을 가상으로 혼합하여 불용성 물질이 생성되는 "
        "침전 반응을 관찰합니다."
    )

    st.info(
        "교육용 반응 모델: Ca²⁺ + CO₃²⁻ → CaCO₃(s)"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("용액 A · CaCl₂")

        ca_M = st.number_input(
            "CaCl₂ 농도 (mol/L)",
            min_value=0.001,
            max_value=2.0,
            value=0.100,
            step=0.01,
            key="ca_M"
        )

        ca_ml = st.number_input(
            "CaCl₂ 부피 (mL)",
            min_value=1.0,
            max_value=500.0,
            value=50.0,
            step=1.0,
            key="ca_ml"
        )

    with c2:

        st.subheader("용액 B · Na₂CO₃")

        co3_M = st.number_input(
            "Na₂CO₃ 농도 (mol/L)",
            min_value=0.001,
            max_value=2.0,
            value=0.100,
            step=0.01,
            key="co3_M"
        )

        co3_add = st.slider(
            "첨가할 Na₂CO₃ 부피 (mL)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key="co3_add"
        )

    precip_moles, remaining_ca, remaining_co3 = precipitation_amount(
        ca_M,
        ca_ml,
        co3_M,
        co3_add
    )

    total_volume = ca_ml + co3_add

    st.divider()

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "생성된 CaCO₃",
        f"{precip_moles:.5f} mol"
    )

    r2.metric(
        "총 용액 부피",
        f"{total_volume:.1f} mL"
    )

    r3.metric(
        "침전 생성 여부",
        "생성됨" if precip_moles > 0 else "없음"
    )

    st.subheader("🫙 실시간 침전 상태")

    precip_height = min(
        80,
        precip_moles * 100000
    )

    liquid_height = 65

    st.markdown(
        f"""
        <div class="beaker">

            <div class="liquid"
                 style="
                 height:{liquid_height}%;
                 background-color:#dbeafe;
                 opacity:0.8;">
            </div>

            <div class="precipitate"
                 style="
                 height:{precip_height}px;">
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if precip_moles > 0:

        st.success(
            f"⚗️ CaCO₃ 침전이 생성되었습니다. "
            f"생성량: {precip_moles:.5f} mol"
        )

    else:

        st.info(
            "현재 조건에서는 아직 침전이 생성되지 않았습니다."
        )

    # 농도 변화

    if total_volume > 0:

        ca_remaining_M = remaining_ca / (
            total_volume / 1000
        )

        co3_remaining_M = remaining_co3 / (
            total_volume / 1000
        )

    else:

        ca_remaining_M = 0
        co3_remaining_M = 0

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Ca²⁺", "CO₃²⁻", "CaCO₃(s)"],
            y=[
                ca_remaining_M,
                co3_remaining_M,
                precip_moles
            ],
            name="현재 상태"
        )
    )

    fig.update_layout(
        title="반응 후 물질의 상대적 양",
        yaxis_title="양 / 농도",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if st.button("💾 침전 실험 결과 저장"):

        save_record(
            "침전 실험",
            "침전 반응",
            {
                "CaCl2 농도": ca_M,
                "CaCl2 부피": ca_ml,
                "Na2CO3 농도": co3_M,
                "Na2CO3 첨가량": co3_add
            },
            {
                "CaCO3 생성량": precip_moles,
                "남은 Ca2+": remaining_ca,
                "남은 CO3^2-": remaining_co3
            }
        )

        st.success("침전 실험 결과가 저장되었습니다.")


# ============================================================
# 화학 계산기
# ============================================================

elif menu == "🧮 화학 계산기":

    st.header("🧮 화학 계산 센터")

    st.write(
        "가화실에서 사용하는 주요 화학 계산식을 "
        "독립적으로 확인할 수 있습니다."
    )

    calculator = st.selectbox(
        "계산 종류",
        [
            "몰농도",
            "몰수",
            "희석",
            "강산 pH",
            "강염기 pH",
            "중화 후 pH"
        ]
    )

    st.divider()

    if calculator == "몰농도":

        n = st.number_input(
            "몰수 n (mol)",
            min_value=0.000001,
            value=0.1
        )

        V = st.number_input(
            "부피 V (mL)",
            min_value=0.001,
            value=100.0
        )

        result = calculate_molarity(n, V)

        st.metric(
            "몰농도",
            f"{result:.6f} mol/L"
        )

        st.latex(r"M=\frac{n}{V}")

    elif calculator == "몰수":

        M = st.number_input(
            "몰농도 M (mol/L)",
            min_value=0.000001,
            value=0.1
        )

        V = st.number_input(
            "부피 V (mL)",
            min_value=0.001,
            value=100.0
        )

        result = calculate_moles(M, V)

        st.metric(
            "몰수",
            f"{result:.6f} mol"
        )

        st.latex(r"n=MV")

    elif calculator == "희석":

        M1 = st.number_input(
            "M₁",
            min_value=0.000001,
            value=1.0
        )

        V1 = st.number_input(
            "V₁ (mL)",
            min_value=0.001,
            value=10.0
        )

        V2 = st.number_input(
            "V₂ (mL)",
            min_value=0.001,
            value=100.0
        )

        result = calculate_dilution(
            M1,
            V1,
            V2
        )

        st.metric(
            "M₂",
            f"{result:.6f} mol/L"
        )

        st.latex(
            r"M_1V_1=M_2V_2"
        )

    elif calculator == "강산 pH":

        M = st.number_input(
            "[H⁺] (mol/L)",
            min_value=0.0000001,
            value=0.1
        )

        pH = -safe_log10(M)

        st.metric(
            "pH",
            f"{pH:.4f}"
        )

        st.latex(
            r"pH=-\log[H^+]"
        )

    elif calculator == "강염기 pH":

        M = st.number_input(
            "[OH⁻] (mol/L)",
            min_value=0.0000001,
            value=0.1
        )

        pOH = -safe_log10(M)
        pH = 14 - pOH

        st.metric(
            "pH",
            f"{pH:.4f}"
        )

        st.latex(
            r"pOH=-\log[OH^-]"
        )

        st.latex(
            r"pH+pOH=14"
        )

    elif calculator == "중화 후 pH":

        acid_M = st.number_input(
            "산 농도",
            min_value=0.000001,
            value=0.1
        )

        acid_V = st.number_input(
            "산 부피 (mL)",
            min_value=0.001,
            value=50.0
        )

        base_M = st.number_input(
            "염기 농도",
            min_value=0.000001,
            value=0.1
        )

        base_V = st.number_input(
            "염기 부피 (mL)",
            min_value=0.0,
            value=50.0
        )

        pH = neutralization_ph(
            acid_M,
            acid_V,
            base_M,
            base_V
        )

        st.metric(
            "중화 후 예상 pH",
            f"{pH:.4f}"
        )


# ============================================================
# 실험 기록
# ============================================================

elif menu == "📚 실험 기록":

    st.header("📚 실험 기록")

    if len(st.session_state.records) == 0:

        st.info(
            "아직 저장된 실험 결과가 없습니다."
        )

    else:

        st.write(
            f"현재 저장된 실험: "
            f"**{len(st.session_state.records)}개**"
        )

        rows = []

        for record in st.session_state.records:

            result = record["결과"]

            rows.append(
                {
                    "시간": record["시간"],
                    "실험": record["실험"],
                    "핵심 결과": json.dumps(
                        result,
                        ensure_ascii=False
                    )
                }
            )

        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            use_container_width=True
        )

        st.divider()

        st.subheader("📊 실험 결과 비교")

        selected = st.multiselect(
            "비교할 실험",
            options=list(range(len(st.session_state.records))),
            format_func=lambda x:
                f"{x+1}. {st.session_state.records[x]['실험']} "
                f"({st.session_state.records[x]['시간']})"
        )

        if len(selected) >= 1:

            comparison = []

            for idx in selected:

                record = st.session_state.records[idx]

                result = record["결과"]

                for key, value in result.items():

                    if isinstance(value, (int, float)):

                        comparison.append(
                            {
                                "실험": record["실험"],
                                "항목": key,
                                "값": value
                            }
                        )

            if comparison:

                comparison_df = pd.DataFrame(
                    comparison
                )

                st.dataframe(
                    comparison_df,
                    use_container_width=True
                )

                numeric_items = comparison_df[
                    comparison_df["값"].apply(
                        lambda x:
                        isinstance(x, (int, float))
                    )
                ]

                if not numeric_items.empty:

                    fig = px.bar(
                        numeric_items,
                        x="실험",
                        y="값",
                        color="항목",
                        barmode="group",
                        title="실험 결과 비교"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

        st.divider()

        if st.button(
            "🗑️ 전체 실험 기록 삭제"
        ):

            st.session_state.records = []

            st.success(
                "실험 기록이 삭제되었습니다."
            )

            st.rerun()


# ============================================================
# 하단 안내
# ============================================================

st.divider()

st.caption(
    "🧪 가화실 · 가상 화학 실험실 | "
    "교육용 시뮬레이션 | 실제 화학 실험을 대체하지 않습니다."
)
