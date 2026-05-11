import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# 1. 页面标题
st.set_page_config(page_title="Employee Attrition Risk Predictor", layout="wide")
st.title("👩‍💼 Employee Attrition Risk Prediction System")
st.subheader("Input Employee Information → Get Attrition Probability & Risk Level")

# 2. 加载数据+训练模型（后台自动跑）
@st.cache_resource
def load_model():
    df = pd.read_csv("IBM HR Analytics Employee Attrition.csv")
    features = ['Age', 'BusinessTravel', 'Department', 'JobRole', 'OverTime',
                'DistanceFromHome', 'JobSatisfaction', 'WorkLifeBalance',
                'YearsAtCompany', 'YearsSinceLastPromotion', 'StockOptionLevel']
    X = df[features]
    y = df['Attrition']
    
    # 编码
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    cat_cols = ['BusinessTravel', 'Department', 'JobRole', 'OverTime']
    encoders = {col: LabelEncoder().fit(X[col]) for col in cat_cols}
    for col in cat_cols:
        X[col] = encoders[col].transform(X[col])
    
    # ✅ 新增：训练集测试集分离（关键优化）
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ 优化后的随机森林
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',  # 应对数据不平衡
        random_state=42
    )
    model.fit(X_train, y_train)
    return model, encoders

model, encoders = load_model()

# 3. 表单输入（可视化界面）
with st.form("employee_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=60, value=28)
        travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    with col2:
        role = st.selectbox("Job Role", ["Sales Representative", "Research Scientist", "Laboratory Technician", "Manager", "Other"])
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        distance = st.number_input("Distance From Home", min_value=1, max_value=50, value=10)
    with col3:
        job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 2)
        wlb = st.slider("Work-Life Balance (1-4)", 1, 4, 1)
        years_company = st.number_input("Years At Company", min_value=0, max_value=20, value=1)
        years_promo = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=0)
        stock = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    
    submit = st.form_submit_button("🔍 Predict Attrition Risk", type="primary")

# 4. 预测+展示结果
if submit:
    # 构造输入数据
    input_data = pd.DataFrame({
        'Age': [age], 'BusinessTravel': [travel], 'Department': [dept],
        'JobRole': [role], 'OverTime': [overtime], 'DistanceFromHome': [distance],
        'JobSatisfaction': [job_sat], 'WorkLifeBalance': [wlb],
        'YearsAtCompany': [years_company], 'YearsSinceLastPromotion': [years_promo],
        'StockOptionLevel': [stock]
    })
    
    # 编码
    for col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])
    
    # 预测概率
    prob = model.predict_proba(input_data)[0][1]
    prob_pct = round(prob * 100, 2)
    
    # 风险等级
    if prob >= 0.7:
        risk_level = "🔴 HIGH RISK"
        color = "red"
    elif prob >= 0.3:
        risk_level = "🟡 MEDIUM RISK"
        color = "orange"
    else:
        risk_level = "🟢 LOW RISK"
        color = "green"
    
    # 展示结果卡片
    st.divider()
    st.subheader("📊 Prediction Result")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric("Attrition Probability", f"{prob_pct}%")
    with col_res2:
        st.markdown(f"<h3 style='color:{color}'>{risk_level}</h3>", unsafe_allow_html=True)
    
    # 关键风险因子（TOP3）
    st.subheader("⚠️ Key Risk Factors")
    feature_importance = pd.DataFrame({
        'Feature': ['Age', 'BusinessTravel', 'Department', 'JobRole', 'OverTime',
                    'DistanceFromHome', 'JobSatisfaction', 'WorkLifeBalance',
                    'YearsAtCompany', 'YearsSinceLastPromotion', 'StockOptionLevel'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(3)
    st.bar_chart(feature_importance, x='Feature', y='Importance')
