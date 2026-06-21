import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# =========================
# 页面设置
# =========================
st.set_page_config(
    page_title="T2DM Death Prediction",
    layout="centered"
)

st.title("T2DM Death Prediction Model")
st.write("Enter the following clinical feature values to estimate death risk.")

# =========================
# 加载模型
# =========================
@st.cache_resource
def load_model():
    return joblib.load("rf.pkl")

model = load_model()

# =========================
# 特征范围定义
# 注意：这里的特征顺序必须和你训练 rf.pkl 时的特征顺序一致
# =========================
feature_ranges = {
    "cGAMP": {"min": 0.0, "max": 5000.0, "default": 1379.204, "step": 1.0},
    "sCr": {"min": 0.0, "max": 1000.0, "default": 1.74, "step": 0.01},
    "EF": {"min": 0.0, "max": 100.0, "default": 49.0, "step": 1.0},
    "TyG": {"min": 0.0, "max": 1000.0, "default": 9.842, "step": 0.001},
    "HB": {"min": 0.0, "max": 1000.0, "default": 12.5, "step": 0.1},
    "vldl": {"min": 0.0, "max": 1000.0, "default": 44.0, "step": 0.1},
    "BMI": {"min": 0.0, "max": 1000.0, "default": 31.53, "step": 0.01},
}

# =========================
# 动态生成输入框
# =========================
st.header("Input Features")

input_features = {}

col1, col2 = st.columns(2)

for i, (feature, properties) in enumerate(feature_ranges.items()):
    with col1 if i % 2 == 0 else col2:
        input_features[feature] = st.number_input(
            label=feature,
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            step=float(properties["step"])
        )

# 转换为 DataFrame
input_data = pd.DataFrame([input_features])

st.subheader("Current Input Data")
st.dataframe(input_data)

# =========================
# 预测与 SHAP 可视化
# =========================
if st.button("Predict"):

    # 模型预测
    predicted_class = model.predict(input_data)[0]
    predicted_proba = model.predict_proba(input_data)[0]

    # 找到阳性类别的概率
    # 一般二分类中，1 代表死亡/事件发生
    if hasattr(model, "classes_"):
        class_labels = list(model.classes_)

        if 1 in class_labels:
            positive_class_index = class_labels.index(1)
        else:
            positive_class_index = int(np.argmax(predicted_proba))
    else:
        positive_class_index = 1

    death_probability = predicted_proba[positive_class_index] * 100

    # =========================
    # 显示预测结果
    # =========================
    st.subheader("Prediction Result")

    st.write(f"Predicted class: **{predicted_class}**")
    st.write(f"Predicted probability of death: **{death_probability:.2f}%**")

    fig, ax = plt.subplots(figsize=(8, 1.5))
    text = f"Predicted probability of death: {death_probability:.2f}%"

    ax.text(
        0.5,
        0.5,
        text,
        fontsize=16,
        ha="center",
        va="center",
        transform=ax.transAxes
    )

    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    # =========================
    # SHAP 解释
    # =========================
    st.subheader("SHAP Force Plot")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # 兼容不同版本 SHAP 的输出格式
    if isinstance(shap_values, list):
        # 旧版本 SHAP：二分类通常返回 [class0, class1]
        shap_values_for_class = shap_values[positive_class_index]
        expected_value = explainer.expected_value[positive_class_index]
    else:
        # 新版本 SHAP：可能是三维数组，形状为 samples × features × classes
        if len(shap_values.shape) == 3:
            shap_values_for_class = shap_values[:, :, positive_class_index]
            expected_value = explainer.expected_value[positive_class_index]
        else:
            shap_values_for_class = shap_values
            expected_value = explainer.expected_value

    # 绘制 SHAP force plot
    shap.force_plot(
        expected_value,
        shap_values_for_class[0],
        input_data.iloc[0],
        matplotlib=True,
        show=False
    )

    shap_fig = plt.gcf()
    st.pyplot(shap_fig)
    plt.close(shap_fig)