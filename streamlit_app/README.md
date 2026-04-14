# Walmart 機率性需求預測系統 — Streamlit UI

**基於梯度提升之機率性分位數需求預測系統**  
國立金門大學 工業工程與管理學系 畢業專題

---

## 快速啟動

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 資料說明

將 Colab 訓練產出的 `predictions.csv` 放入 `data/` 資料夾：

```
streamlit_app/
├── app.py
├── requirements.txt
└── data/
    └── predictions.csv   ← 從 Google Drive 下載後放這裡
```

`predictions.csv` 必要欄位：

| 欄位 | 說明 |
|------|------|
| Store, Dept, Date | 識別欄位 |
| Weekly_Sales | 實際銷售額（測試期） |
| IsHoliday | 節慶週標記（0/1） |
| XYZ | 需求分群（X/Y/Z） |
| P10_HW, P50_HW, P90_HW | STL+HW 分位數預測 |
| P10_LGBM, P50_LGBM, P90_LGBM | LightGBM-Q 分位數預測 |

> 若 `data/predictions.csv` 不存在，系統自動載入內建 Demo 資料，不會報錯。

---

## 四頁功能說明

| 頁面 | 功能 |
|------|------|
| 📊 資料探索 | KPI 卡、週銷售趨勢、XYZ 分布、節慶效應 |
| 🔮 預測結果 | 兩種模式：單一時序深度分析 / XYZ 三群並排比較 |
| 📈 模型對照 | 指標切換、雷達圖、完整對照表、假設驗證結論 |
| ℹ️ 研究說明 | 研究架構、方法論、未來補貨決策系統規劃 |

---

## Streamlit Cloud 部署

1. 推送至 GitHub
2. 前往 [share.streamlit.io](https://share.streamlit.io) → New app
3. Repository: `your-account/walmart-capstone`
4. Branch: `main`
5. Main file path: `streamlit_app/app.py`
6. 點 **Deploy**

---

*國立金門大學 工業工程與管理學系 ‧ 2024 畢業專題*
