from pathlib import Path
import math

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 기본 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="계절에 따른 배추 가격",
    page_icon="🥬",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "배추_기상_전처리완료.csv"

REQUIRED_COLUMNS = [
    "연도", "월", "연월", "계절", "배추월평균가격",
    "평균기온", "강수량합계", "일조시간합계",
    "가격자료수", "평균일별품목수",
]


# --------------------------------------------------
# 데이터 불러오기 및 확인
# --------------------------------------------------
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, encoding="utf-8-sig")
    return data


st.title("🥬 계절에 따른 배추 가격")
st.caption("계절과 기상 조건에 따라 배추의 월평균 가격이 어떻게 달라지는지 살펴보고, 선형회귀 모델로 가격을 예측합니다.")

if not DATA_PATH.exists():
    st.error(
        "전처리 완료 CSV 파일을 찾을 수 없습니다.\n\n"
        "main.py와 같은 폴더에 '배추_기상_전처리완료.csv' 파일을 넣어 주세요."
    )
    st.stop()

try:
    df = load_data(DATA_PATH)
except UnicodeDecodeError:
    st.error("CSV 인코딩을 읽을 수 없습니다. 파일이 UTF-8-SIG 인코딩인지 확인해 주세요.")
    st.stop()
except Exception as error:
    st.error(f"CSV 파일을 불러오는 중 오류가 발생했습니다: {error}")
    st.stop()

missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
if missing_columns:
    st.error(
        "CSV 파일에 필요한 열 이름이 없습니다.\n\n"
        f"누락된 열: {', '.join(missing_columns)}\n\n"
        f"현재 파일의 열 이름: {', '.join(map(str, df.columns))}"
    )
    st.stop()

if df.empty:
    st.error("CSV 파일에 데이터가 없습니다.")
    st.stop()

# 모델과 그래프에 필요한 숫자형 열을 숫자로 변환
numeric_columns = [
    "연도", "월", "배추월평균가격", "평균기온",
    "강수량합계", "일조시간합계",
]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# 모델에 필요한 열의 결측치 제거
model_columns = [
    "연도", "월", "계절", "배추월평균가격",
    "평균기온", "강수량합계", "일조시간합계",
]
before_rows = len(df)
df = df.dropna(subset=model_columns).copy()

if df.empty:
    st.error("모델 학습에 필요한 값이 모두 비어 있거나 숫자로 변환할 수 없습니다.")
    st.stop()

df = df.sort_values(["연도", "월"]).reset_index(drop=True)

if len(df) < 10:
    st.error("모델을 학습하고 평가하기에 데이터가 너무 적습니다. 최소 10개월 이상의 데이터가 필요합니다.")
    st.stop()

if before_rows > len(df):
    st.warning(
        f"모델에 필요한 값이 없는 {before_rows - len(df)}개 행을 분석에서 제외했습니다."
    )


# --------------------------------------------------
# 데이터 소개 및 요약 카드
# --------------------------------------------------
st.subheader("데이터 소개")
st.write(
    "이 데이터는 2018년부터 2024년까지의 월별 배추 가격과 기상 정보를 결합한 자료입니다. "
    "평균기온, 강수량, 일조시간, 계절을 입력 변수로 사용하고 "
    "배추월평균가격을 예측 대상으로 설정했습니다."
)

start_period = str(df.iloc[0]["연월"])
end_period = str(df.iloc[-1]["연월"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("데이터 기간", f"{start_period} ~ {end_period}")
col2.metric("평균기온", f"{df['평균기온'].mean():.1f} °C")
col3.metric("평균 강수량", f"{df['강수량합계'].mean():.1f} mm")
col4.metric("평균 일조시간", f"{df['일조시간합계'].mean():.1f} 시간")

st.divider()


# --------------------------------------------------
# 처음 다섯 줄
# --------------------------------------------------
st.subheader("전처리 데이터 미리보기")
st.write("전체 데이터 중 처음 다섯 줄입니다.")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

st.divider()


# --------------------------------------------------
# 선형회귀 모델 학습
# 시간 순서를 유지하여 앞의 80%는 학습, 뒤의 20%는 테스트
# --------------------------------------------------
st.subheader("선형회귀 모델을 이용한 배추 가격 예측")
st.write(
    "데이터를 연월 순서로 정렬한 뒤 과거 데이터 약 80%로 학습하고, "
    "이후 데이터 약 20%로 예측 성능을 평가합니다. "
    "계절은 원-핫 인코딩으로 숫자형 변수로 변환합니다."
)

feature_columns = ["평균기온", "강수량합계", "일조시간합계", "계절"]
target_column = "배추월평균가격"

X = df[feature_columns]
y = df[target_column]

split_index = int(len(df) * 0.8)
split_index = max(1, min(split_index, len(df) - 1))

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
test_periods = df["연월"].iloc[split_index:]

numeric_features = ["평균기온", "강수량합계", "일조시간합계"]
categorical_features = ["계절"]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        ("season", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

try:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
except Exception as error:
    st.error(f"모델 학습 또는 예측 중 오류가 발생했습니다: {error}")
    st.stop()

mae = mean_absolute_error(y_test, predictions)
rmse = math.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else float("nan")

metric1, metric2, metric3 = st.columns(3)
metric1.metric("MAE", f"{mae:,.1f}원")
metric2.metric("RMSE", f"{rmse:,.1f}원")
metric3.metric("R²", f"{r2:.3f}" if not math.isnan(r2) else "계산 불가")

st.caption(
    "MAE와 RMSE는 예측 가격과 실제 가격의 차이를 원 단위로 나타냅니다. "
    "R²는 모델이 테스트 데이터의 가격 변동을 얼마나 설명하는지 나타내며, "
    "음수일 수도 있습니다."
)

# 실제값과 예측값을 그래프로 비교
comparison_df = pd.DataFrame(
    {
        "연월": test_periods.to_numpy(),
        "실제 가격": y_test.to_numpy(),
        "예측 가격": predictions,
    }
)

chart_df = comparison_df.melt(
    id_vars="연월",
    value_vars=["실제 가격", "예측 가격"],
    var_name="구분",
    value_name="가격(원)",
)

fig = px.line(
    chart_df,
    x="연월",
    y="가격(원)",
    color="구분",
    markers=True,
    title="테스트 데이터의 실제 배추 가격과 예측 가격 비교",
    labels={"연월": "연월", "가격(원)": "배추 가격(원)", "구분": "가격 구분"},
)
fig.update_layout(
    hovermode="x unified",
    legend_title_text="",
    xaxis=dict(type="category"),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("테스트 데이터의 실제 가격과 예측 가격 확인"):
    st.dataframe(comparison_df.round({"실제 가격": 1, "예측 가격": 1}), use_container_width=True, hide_index=True)

st.divider()


# --------------------------------------------------
# 데이터 출처
# --------------------------------------------------
st.subheader("데이터 출처")
st.write(
    "- 기상 자료: 업로드한 ASOS 월별 기상 데이터\n"
    "- 배추 가격 자료: 업로드한 농수축산물 도소매 가격 데이터\n"
)
st.caption(
    "※ 제출 전 원본 자료에서 제공기관명과 원본 URL을 확인해 위 출처에 구체적으로 기재하세요."
)

