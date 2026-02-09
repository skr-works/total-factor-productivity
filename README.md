# TFP

> 注意：ここでのTFPは、労働投入・資本ストック・物価調整を用いた厳密なTFP推計ではありません。会計データの範囲で「効率改善の兆候」を推定するものです。

---

## 目的

- 企業の「資本に対するアウトプットの伸び（資本生産性の改善）」、一次スクリーニング用途。
  
---

## 入力 / 出力

### 入力
- 銘柄コード（例：日本株は `XXXX.T` 形式に変換して取得）

### 出力（1銘柄あたり）
- 総合判定：AA / A / B / C / D
- 年次スコア（0–100）
- データ品質（0–100）
- 警告フラグ（Yellow / Red）
- 判定理由（キャップ・降格理由）
- 決算期末 / 採用Output / 採用Capital
- 年次: 生産性成長（%）
- 年次: 利益率トレンド（改善 / 維持 / 悪化）
- 年次: 安定回数（増益年数カウント）
- 成長の質（QualityGap, pt）

---

## データソース（yfinance）

- 年次：income_stmt / balance_sheet（必須）
- 四半期：quarterly_income_stmt（任意、参考情報）
- 取得できない科目は欠損扱いとし、品質ゲート・キャップに反映する

---

## 指標定義（年次ベース）

### 1) Capital Productivity（資本生産性）
- `CP(t) = Output(t) / Capital(t)`
- `CP_CAGR`：CPの年平均成長率（可能なら5年、無理なら短縮）

### 2) 利益率（マージン）
- 利益率は `Operating Income / Revenue` を優先
- 取得不能な場合は `EBITDA / Revenue` にフォールバック
- 「Output/Revenue」を利益率として扱うことは禁止（定義が崩れるため）

### 3) 成長の質（QualityGap）
- `QualityGap = CAGR(Output) - CAGR(Capital)`（pt表示）
- 資産増だけで伸びたケースと、効率で伸びたケースを分離するための中核指標

---

## スコア（0–100）と判定（AA/A/B/C/D）

### スコア構成（合計100）
- CP_CAGR（資本生産性の伸び）：50
- ΔMargin（利益率の改善幅）：30
- QualityGap（成長の質）：20

### 判定（例）
- AA / A / B / C / D を閾値で分類  
（閾値は「Aが量産されない」よう厳しめに設定）

---

## 過信防止（重要）

### データ品質ゲート
- `DataQualityScore < 40`：D固定
- `DataQualityScore < 60`：上限C

### 代表的なフラグ
- Red: Deficit（最新期のOutputがマイナス）
- Red: RevBase（OutputがTotal Revenue採用）
- Red: Struct_Assets / Struct_Revenue（資産・売上の急変＝構造変化疑い）
- Red: Outlier（異常値＝暴発）
- Yellow: Annual_Only（四半期データが取れない）

### キャップ（上限規制）の考え方
- Redがある場合、AA/Aを許可しない（内容によってはD固定）
- QualityGapが欠損（NA）の場合、AA/A/Bを許可しない（最大C）
- Annual_Onlyは高評価を抑制（最大Bなど）

---

## 使い方（運用上の推奨）

- 上位グレードから順に見るのではなく、まず **Redフラグの有無**で足切りする
- 次に **DataQualityScore** が低い銘柄を外す
- 最後に **QualityGap（成長の質）** を重視して候補を絞る

---


