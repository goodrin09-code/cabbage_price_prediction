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
# 1. 페이지 설정
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

NUMERIC_COLUMNS = [
    "연도", "월", "배추월평균가격", "평균기온",
    "강수량합계", "일조시간합계", "가격자료수", "평균일별품목수",
]

MODEL_COLUMNS = [
    "연도", "월", "계절", "배추월평균가격",
    "평균기온", "강수량합계", "일조시간합계",
]

SEASON_ORDER = ["봄", "여름", "가을", "겨울"]
SEASON_COLORS = {
    "봄": "#82B366",
    "여름": "#F2B84B",
    "가을": "#D98B45",
    "겨울": "#6C9BCF",
}


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, encoding="utf-8-sig")


st.title("🥬 계절에 따른 배추 가격")
st.caption(
    "계절과 기상 조건에 따라 배추의 월평균 가격이 어떻게 달라지는지 분석하고, "
    "선형회귀 모델을 이용해 가격을 추정하는 데이터과학 탐구입니다."
)

if not DATA_PATH.exists():
    st.error(
        "전처리 완료 CSV 파일을 찾을 수 없습니다.\n\n"
        "main.py와 같은 폴더에 '배추_기상_전처리완료.csv' 파일을 넣어 주세요."
    )
    st.stop()

try:
    raw_df = load_data(DATA_PATH)
except UnicodeDecodeError:
    st.error("CSV 인코딩 오류입니다. 파일이 UTF-8-SIG 인코딩인지 확인해 주세요.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("CSV 파일에 데이터가 없습니다.")
    st.stop()
except Exception as error:
    st.error(f"CSV 파일을 불러오는 중 오류가 발생했습니다: {error}")
    st.stop()

if raw_df.empty:
    st.error("CSV 파일에 데이터가 없습니다.")
    st.stop()

missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw_df.columns]
if missing_columns:
    st.error(
        "CSV 파일에 필요한 열 이름이 없습니다.\n\n"
        f"누락된 열: {', '.join(missing_columns)}\n\n"
        f"현재 파일의 열 이름: {', '.join(map(str, raw_df.columns))}"
    )
    st.stop()

# 원본 데이터의 처음 다섯 줄을 보존
preview_df = raw_df.head(5).copy()

# 분석용 데이터 별도 생성
df = raw_df.copy()

for column in NUMERIC_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["계절"] = df["계절"].astype("string").str.strip()
df["연월"] = df["연월"].astype("string").str.strip()

# 모델에 필요한 값이 없는 행은 모델 분석에서 제외
before_cleaning = len(df)
df = df.dropna(subset=MODEL_COLUMNS).copy()
removed_rows = before_cleaning - len(df)

if df.empty:
    st.error(
        "모델 분석에 사용할 수 있는 데이터가 없습니다. "
        "가격·기상·연도·월·계절 열의 값과 자료형을 확인해 주세요."
    )
    st.stop()

# 연도와 월의 유효성 확인
invalid_month = ~df["월"].between(1, 12)
invalid_year = df["연도"] < 1

if (invalid_month | invalid_year).any():
    st.error("연도 또는 월에 올바르지 않은 값이 있습니다. 월은 1~12 사이여야 합니다.")
    st.stop()

# 계절 값 확인
unknown_seasons = sorted(set(df["계절"].dropna()) - set(SEASON_ORDER))
if unknown_seasons:
    st.error(
        "계절 열에 예상하지 못한 값이 있습니다.\n\n"
        f"확인된 값: {', '.join(unknown_seasons)}\n"
        "계절 값은 봄, 여름, 가을, 겨울 중 하나로 통일해 주세요."
    )
    st.stop()

# 연도와 월 기준으로 정렬
df = df.sort_values(["연도", "월"]).reset_index(drop=True)

# 같은 연월이 두 번 이상 있으면 월별 분석 및 시간 순서 평가가 왜곡될 수 있음
duplicate_mask = df.duplicated(subset=["연도", "월"], keep=False)
if duplicate_mask.any():
    duplicate_months = (
        df.loc[duplicate_mask, ["연도", "월"]]
        .drop_duplicates()
        .assign(연월=lambda x: x["연도"].astype(int).astype(str)
                + "-"
                + x["월"].astype(int).astype(str).str.zfill(2))
    )
    duplicate_text = ", ".join(duplicate_months["연월"].tolist()[:10])
    st.error(
        "동일한 연도·월 데이터가 중복되어 있습니다. "
        "월별 분석 전에 중복 자료를 확인해 주세요.\n\n"
        f"중복된 연월 예시: {duplicate_text}"
    )
    st.stop()

if len(df) < 10:
    st.error("학습과 평가를 진행하기에 데이터가 부족합니다. 최소 10개월 이상의 데이터가 필요합니다.")
    st.stop()

if removed_rows > 0:
    st.warning(
        f"모델 분석에 필요한 값이 없는 {removed_rows}개 행을 분석에서 제외했습니다. "
        "아래 데이터 요약과 그래프는 유효한 분석 데이터 기준입니다."
    )

# 그래프에서 사용할 연월 표시를 연도·월로 다시 생성해 형식 통일
df["연월_표시"] = (
    df["연도"].astype(int).astype(str)
    + "-"
    + df["월"].astype(int).astype(str).str.zfill(2)
)


# --------------------------------------------------
# 3. 데이터 소개
# --------------------------------------------------
st.header("1. 데이터 소개")
st.write(
    "이 데이터는 월별 배추 가격과 기상 정보를 결합한 자료입니다. "
    "계절별 가격 차이를 살펴보고, 평균기온·강수량·일조시간이 배추 가격과 "
    "어떤 관계가 있는지 분석합니다."
)

start_period = df.iloc[0]["연월_표시"]
end_period = df.iloc[-1]["연월_표시"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("데이터 기간", f"{start_period} ~ {end_period}")
c2.metric("평균기온", f"{df['평균기온'].mean():.1f} °C")
c3.metric("평균 강수량", f"{df['강수량합계'].mean():.1f} mm")
c4.metric("평균 일조시간", f"{df['일조시간합계'].mean():.1f} 시간")

st.caption(
    f"원본 데이터 {len(raw_df):,}행 · 분석에 사용한 데이터 {len(df):,}행 · "
    f"제외된 데이터 {removed_rows:,}행"
)

with st.expander("전처리 데이터 미리보기"):
    st.write("CSV 원본 데이터의 처음 다섯 줄입니다.")
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

st.divider()


# --------------------------------------------------
# 4. 전체 배추 가격의 월별 변화
# --------------------------------------------------
st.header("2. 월별 배추 가격 변화")
st.write(
    "시간의 흐름에 따라 배추 월평균 가격이 어떻게 변했는지 확인합니다. "
    "가격이 특정 시기에 크게 오르거나 내리는 구간이 있는지 살펴보세요."
)

price_fig = px.line(
    df,
    x="연월_표시",
    y="배추월평균가격",
    markers=True,
    labels={"연월_표시": "연월", "배추월평균가격": "배추 월평균 가격(원)"},
    title="월별 배추 평균 가격 추이",
)
price_fig.update_layout(hovermode="x unified")
st.plotly_chart(price_fig, use_container_width=True)


# --------------------------------------------------
# 5. 계절별 가격 분석
# --------------------------------------------------
st.header("3. 계절별 배추 가격 비교")
st.write(
    "전체 분석 기간의 자료를 계절별로 묶어 평균 가격을 비교합니다. "
    "이를 통해 배추 가격이 계절에 따라 차이가 있는지 살펴볼 수 있습니다."
)

season_summary = (
    df.groupby("계절", as_index=False)
    .agg(
        평균가격=("배추월평균가격", "mean"),
        중앙값=("배추월평균가격", "median"),
        자료수=("배추월평균가격", "count"),
    )
)
season_summary["계절"] = pd.Categorical(
    season_summary["계절"], categories=SEASON_ORDER, ordered=True
)
season_summary = season_summary.sort_values("계절")

season_fig = px.bar(
    season_summary,
    x="계절",
    y="평균가격",
    color="계절",
    category_orders={"계절": SEASON_ORDER},
    color_discrete_map=SEASON_COLORS,
    text=season_summary["평균가격"].map(lambda value: f"{value:,.0f}원"),
    labels={"계절": "계절", "평균가격": "평균 배추 가격(원)"},
    title="계절별 배추 월평균 가격",
)
season_fig.update_traces(textposition="outside")
season_fig.update_layout(showlegend=False)
st.plotly_chart(season_fig, use_container_width=True)

st.dataframe(
    season_summary.assign(
        평균가격=season_summary["평균가격"].map(lambda x: f"{x:,.1f}원"),
        중앙값=season_summary["중앙값"].map(lambda x: f"{x:,.1f}원"),
    )[["계절", "평균가격", "중앙값", "자료수"]],
    use_container_width=True,
    hide_index=True,
)

if len(season_summary) >= 2:
    highest_season = season_summary.loc[season_summary["평균가격"].idxmax()]
    lowest_season = season_summary.loc[season_summary["평균가격"].idxmin()]
    price_gap = highest_season["평균가격"] - lowest_season["평균가격"]

    st.info(
        f"분석 기간 중 평균 가격이 가장 높은 계절은 **{highest_season['계절']}** "
        f"({highest_season['평균가격']:,.1f}원)이고, "
        f"가장 낮은 계절은 **{lowest_season['계절']}** "
        f"({lowest_season['평균가격']:,.1f}원)입니다. "
        f"두 계절의 평균 가격 차이는 **{price_gap:,.1f}원**입니다."
    )
    st.caption(
        "이 결과는 분석 기간의 평균적인 차이를 나타냅니다. "
        "계절만으로 가격 차이의 원인을 확정할 수는 없습니다."
    )

st.divider()


# --------------------------------------------------
# 6. 기상 조건과 가격의 관계
# --------------------------------------------------
st.header("4. 기상 조건과 배추 가격의 관계")
st.write(
    "기온·강수량·일조시간과 배추 가격의 관계를 산점도와 상관계수로 확인합니다. "
    "상관관계는 두 변수의 함께 변화하는 경향을 나타내며, 인과관계를 의미하지는 않습니다."
)

weather_columns = ["평균기온", "강수량합계", "일조시간합계"]
weather_labels = {
    "평균기온": "평균기온 (°C)",
    "강수량합계": "월 강수량 (mm)",
    "일조시간합계": "월 일조시간",
}

corr_values = {}
for column in weather_columns:
    corr_values[column] = df[column].corr(df["배추월평균가격"])

corr_cols = st.columns(3)
for i, column in enumerate(weather_columns):
    value = corr_values[column]
    corr_cols[i].metric(
        f"{column}과 가격의 상관계수",
        f"{value:.3f}" if pd.notna(value) else "계산 불가",
    )

for column in weather_columns:
    scatter_fig = px.scatter(
        df,
        x=column,
        y="배추월평균가격",
        color="계절",
        category_orders={"계절": SEASON_ORDER},
        color_discrete_map=SEASON_COLORS,
        hover_data=["연월_표시"],
        labels={
            column: weather_labels[column],
            "배추월평균가격": "배추 월평균 가격(원)",
            "계절": "계절",
        },
        title=f"{weather_labels[column]}과 배추 가격의 관계",
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

st.caption(
    "상관계수가 양수이면 한 변수가 커질 때 다른 변수도 커지는 경향, "
    "음수이면 한 변수가 커질 때 다른 변수는 작아지는 경향을 뜻합니다. "
    "상관계수가 0에 가깝더라도 비선형 관계가 존재할 수 있습니다."
)


# --------------------------------------------------
# 7. 선형회귀 모델 학습 및 평가
# --------------------------------------------------
st.divider()
st.header("5. 선형회귀 모델을 이용한 배추 가격 추정")
st.write(
    "연월 순서로 정렬한 뒤 과거 데이터 약 80%를 학습에 사용하고, "
    "그 이후 데이터 약 20%를 테스트에 사용합니다. "
    "계절은 원-핫 인코딩으로 변환하고, 기상 변수와 함께 모델에 입력합니다."
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
test_periods = df["연월_표시"].iloc[split_index:]

numeric_features = ["평균기온", "강수량합계", "일조시간합계"]
categorical_features = ["계절"]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        (
            "season",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
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

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:,.1f}원")
m2.metric("RMSE", f"{rmse:,.1f}원")
m3.metric("R²", f"{r2:.3f}" if not math.isnan(r2) else "계산 불가")

st.markdown(
    """
    - **MAE**: 예측 가격과 실제 가격 차이의 절댓값을 평균한 값입니다. 작을수록 좋습니다.
    - **RMSE**: 큰 오차에 더 큰 영향을 받는 평가 지표입니다. 작을수록 좋습니다.
    - **R²**: 테스트 데이터의 가격 변동을 모델이 얼마나 설명하는지 나타냅니다.
      1에 가까울수록 설명력이 높고, 음수가 나올 수도 있습니다.
    """
)

if not math.isnan(r2):
    if r2 >= 0.7:
        r2_comment = "테스트 데이터의 가격 변동을 비교적 잘 설명하는 결과입니다."
    elif r2 >= 0.4:
        r2_comment = "테스트 데이터의 가격 변동 중 일부를 설명하지만, 설명하지 못하는 변화도 남아 있습니다."
    elif r2 >= 0:
        r2_comment = "테스트 데이터의 가격 변동을 설명하는 정도가 낮은 편입니다."
    else:
        r2_comment = "테스트 데이터에서 평균 가격만으로 예측하는 기준보다 성능이 낮게 나타났습니다."

    st.info(
        f"이번 모델의 R²는 **{r2:.3f}**입니다. {r2_comment} "
        "배추 가격에는 기상 조건과 계절 외에도 생산량, 유통 상황, 수요 등 "
        "여러 요인이 영향을 줄 수 있습니다."
    )

# 실제값·예측값·오차 표
comparison_df = pd.DataFrame(
    {
        "연월": test_periods.to_numpy(),
        "실제 가격(원)": y_test.to_numpy(),
        "예측 가격(원)": predictions,
    }
)
comparison_df["오차(원)"] = (
    comparison_df["예측 가격(원)"] - comparison_df["실제 가격(원)"]
)
comparison_df["절대 오차(원)"] = comparison_df["오차(원)"].abs()

# 실제값과 예측값 비교 그래프
chart_df = comparison_df.melt(
    id_vars="연월",
    value_vars=["실제 가격(원)", "예측 가격(원)"],
    var_name="구분",
    value_name="가격(원)",
)

prediction_fig = px.line(
    chart_df,
    x="연월",
    y="가격(원)",
    color="구분",
    markers=True,
    title="테스트 데이터의 실제 배추 가격과 예측 가격 비교",
    labels={"연월": "연월", "가격(원)": "배추 가격(원)", "구분": "가격 구분"},
)
prediction_fig.update_layout(
    hovermode="x unified",
    legend_title_text="",
    xaxis=dict(type="category"),
)
st.plotly_chart(prediction_fig, use_container_width=True)

st.subheader("예측 오차 확인")
best_row = comparison_df.loc[comparison_df["절대 오차(원)"].idxmin()]
worst_row = comparison_df.loc[comparison_df["절대 오차(원)"].idxmax()]

e1, e2 = st.columns(2)
e1.metric("오차가 가장 작았던 달", best_row["연월"])
e1.caption(f"절대 오차: {best_row['절대 오차(원)']:,.1f}원")
e2.metric("오차가 가장 컸던 달", worst_row["연월"])
e2.caption(f"절대 오차: {worst_row['절대 오차(원)']:,.1f}원")

with st.expander("테스트 데이터의 실제 가격·예측 가격·오차 표 보기"):
    display_comparison = comparison_df.copy()
    for column in ["실제 가격(원)", "예측 가격(원)", "오차(원)", "절대 오차(원)"]:
        display_comparison[column] = display_comparison[column].map(lambda x: f"{x:,.1f}")
    st.dataframe(display_comparison, use_container_width=True, hide_index=True)


# --------------------------------------------------
# 8. 탐구 결과 및 한계
# --------------------------------------------------
st.divider()
st.header("6. 탐구 결과 및 결론")

if len(season_summary) >= 2:
    st.markdown(
        f"""
        **① 계절별 가격 차이**

        분석 기간 동안 계절별 평균 배추 가격을 비교한 결과,
        평균 가격이 가장 높은 계절은 **{highest_season['계절']}**,
        가장 낮은 계절은 **{lowest_season['계절']}**로 나타났습니다.
        두 계절의 평균 가격 차이는 약 **{price_gap:,.1f}원**입니다.
        """
    )

st.markdown(
    """
    **② 기상 조건과 가격의 관계**

    기온·강수량·일조시간과 배추 가격의 상관계수를 비교하여
    어떤 기상 변수가 가격과 상대적으로 강한 선형 관계를 보이는지 살펴볼 수 있습니다.
    상관관계만으로 가격 변화의 원인을 단정할 수는 없습니다.
    """
)

st.markdown(
    f"""
    **③ 선형회귀 모델 평가**

    테스트 데이터에서 MAE는 약 **{mae:,.1f}원**,
    RMSE는 약 **{rmse:,.1f}원**,
    R²는 **{r2:.3f}**로 나타났습니다.
    이 결과는 기상 조건과 계절만으로 배추 가격의 모든 변화를 설명하기 어렵다는 점을 보여줍니다.
    """
)

st.markdown(
    """
    **④ 탐구의 한계와 발전 방향**

    이번 모델은 평균기온·강수량·일조시간·계절만을 사용했습니다.
    실제 배추 가격에는 재배 면적과 생산량, 작황, 유통 비용, 소비 수요,
    명절 등 다양한 요인이 영향을 줄 수 있습니다.
    향후 이러한 자료를 추가하면 가격 변화에 대한 분석과 예측을 개선할 수 있을 것입니다.
    """
)

st.warning(
    "주의: 이 앱의 모델은 미래의 기상 예보를 이용해 아직 발생하지 않은 가격을 예측하는 기능이 아닙니다. "
    "과거 데이터로 학습한 뒤 학습에 포함되지 않은 이후 기간의 실제 기상 자료를 입력하여 "
    "그 기간의 가격을 추정하고 평가하는 방식입니다."
)


# --------------------------------------------------
# 9. 데이터 출처
# --------------------------------------------------
st.divider()
st.header("7. 데이터 출처")
st.write(
    "- 기상 자료: 사용한 ASOS 월별 기상 데이터\n"
    "- 배추 가격 자료: 사용한 농수축산물 도소매 가격 데이터\n"
    "- 분석 기간: 위 CSV 파일에 포함된 전체 기간"
)
st.caption(
    "※ 제출 전 원본 자료에서 실제 제공기관명과 원본 URL을 확인해 위 출처에 구체적으로 기재하세요."
)

st.caption(
    "본 웹앱은 pandas를 이용한 데이터 처리, Plotly를 이용한 시각화, "
    "scikit-learn을 이용한 선형회귀 분석으로 제작했습니다."
)
