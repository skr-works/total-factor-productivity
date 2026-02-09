import os
import json
import time
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- 設定・定数 ---
# ログに銘柄名を出さないための配慮
def log_progress(current, total):
    print(f"[System] Progress: {current}/{total} items processed.")

# --- コアロジック (TFP Proxy Engine) ---

class TFPScorer:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)
        self.data_ok = False
        self.logs = [] # 計算根拠のメモ
        
        # 結果格納用
        self.output_metric_used = "-"
        self.capital_metric_used = "-"
        self.base_score = 0
        self.data_quality_score = 0
        self.flags_yellow = []
        self.flags_red = []
        self.final_grade = "D"
        self.reason = ""
        self.metrics = {} # 数値データ格納

    def run_analysis(self):
        try:
            # 1. データ取得
            # 年次データを取得 (最新が左、過去が右に来るようソート確認)
            income = self.ticker.income_stmt
            bs = self.ticker.balance_sheet
            cf = self.ticker.cashflow
            
            if income.empty or bs.empty:
                self.reason = "財務データ取得不可"
                return

            # 列（日付）を昇順（過去→未来）に並べ替え
            income = income.sort_index(axis=1)
            bs = bs.sort_index(axis=1)
            cf = cf.sort_index(axis=1)
            
            years_count = len(income.columns)
            
            # --- 5.1.1 Q1: 必要期間の充足 ---
            q1_score = 0
            if years_count >= 5: q1_score = 40
            elif years_count == 4: q1_score = 25
            elif years_count == 3: q1_score = 10
            else: q1_score = 0
            
            # --- 1.3 科目マッピング (Output) ---
            output_series = None
            q2_score = 0
            
            # 利用可能な行名を探すヘルパー関数
            def get_row(df, candidates):
                for c in candidates:
                    if c in df.index:
                        return df.loc[c], c
                return None, None

            # Output候補
            output_candidates = ['EBITDA', 'Operating Income', 'Gross Profit', 'Total Revenue']
            output_series, output_name = get_row(income, output_candidates)
            
            if output_name in ['EBITDA', 'Operating Income']: q2_score = 30
            elif output_name == 'Gross Profit': q2_score = 15
            elif output_name == 'Total Revenue': q2_score = 0
            else: q2_score = 0 # データなし
            
            self.output_metric_used = output_name if output_name else "None"

            # Capital候補 (Assets)
            assets_series, assets_name = get_row(bs, ['Total Assets'])
            ppe_series, ppe_name = get_row(bs, ['Property Plant Equipment', 'Net PPE'])
            
            # Capital計算 (平均残高)
            capital_series = pd.Series(index=output_series.index, dtype='float64')
            q3_score = 0
            
            if assets_series is not None:
                # 平均資産計算: (Current + Previous) / 2
                # pandasのshiftを使って計算
                prev_assets = assets_series.shift(1) # 1つ前の期間
                avg_assets = (assets_series + prev_assets) / 2
                
                # 平均が計算できる年だけ残す
                capital_series = avg_assets
                q3_score = 30 # 平均計算OK
                self.capital_metric_used = "AvgAssets"
                
                # もし平均計算で全部NaNになった場合（データが1年分しかない等）
                if capital_series.dropna().empty:
                    capital_series = assets_series # 期末残高で代用
                    q3_score = 15
                    self.capital_metric_used = "AssetsEnd"
            else:
                q3_score = 0
                self.capital_metric_used = "None"

            # Data Quality Score 算出
            self.data_quality_score = q1_score + q2_score + q3_score
            
            # --- ゲートチェック (5.1.2) ---
            # ここでリターンせず、計算できるところまで計算して最後に判定する仕様に従う
            
            # --- 3. 指標定義 ---
            # 共通のインデックスを持つデータフレームを作成して計算
            df_calc = pd.DataFrame({
                'Output': output_series,
                'Capital': capital_series,
                'Revenue': get_row(income, ['Total Revenue'])[0]
            }).dropna()

            if len(df_calc) < 2:
                self.reason = "計算に必要な期間不足"
                self.final_grade = "D"
                return # 計算不能

            # 資本生産性 CP
            df_calc['CP'] = df_calc['Output'] / df_calc['Capital']
            
            # マージン (Operating Incomeがない場合はEBITDA等で代替されているOutputを使う)
            if df_calc['Revenue'] is not None:
                df_calc['Margin'] = df_calc['Output'] / df_calc['Revenue']
            else:
                df_calc['Margin'] = 0

            # CAGR計算ヘルパー
            def calc_cagr(series, n_years):
                if len(series) < n_years: return None
                end_val = series.iloc[-1]
                start_val = series.iloc[-(n_years)] # n年前
                if start_val <= 0 or end_val <= 0: return 0 # 負の値のCAGRは扱わない簡易実装
                return (end_val / start_val) ** (1/(n_years-1)) - 1

            # 直近データを使用
            n_cagr = 5 if len(df_calc) >= 5 else 3
            if len(df_calc) < 3: n_cagr = len(df_calc)
            
            cagr_output = calc_cagr(df_calc['Output'], n_cagr) or 0
            cagr_capital = calc_cagr(df_calc['Capital'], n_cagr) or 0
            
            # 3.3 CP_CAGR (S1用)
            # CP自体のCAGRを見る
            cagr_cp = calc_cagr(df_calc['CP'], n_cagr) or 0
            
            # 3.5 Quality Gap (S3用)
            quality_gap = cagr_output - cagr_capital

            # 3.4 Margin改善 (S2用)
            # 直近3年平均 - その前3年平均
            margin_vals = df_calc['Margin'].values
            if len(margin_vals) >= 6:
                avg_recent = np.mean(margin_vals[-3:])
                avg_past = np.mean(margin_vals[-6:-3])
                delta_margin = avg_recent - avg_past
            elif len(margin_vals) >= 4:
                avg_recent = np.mean(margin_vals[-2:])
                avg_past = np.mean(margin_vals[-4:-2])
                delta_margin = avg_recent - avg_past
            else:
                delta_margin = 0

            # --- 4. スコア仕様 ---
            # S1: 資本生産性の伸び (Max 50)
            s1 = 0
            if cagr_cp >= 0.06: s1 = 50
            elif cagr_cp >= 0.03: s1 = 35
            elif cagr_cp >= 0: s1 = 20
            else: s1 = 0
            
            # S2: マージン改善 (Max 30)
            s2 = 0
            if delta_margin >= 0.03: s2 = 30
            elif delta_margin >= 0.01: s2 = 20
            elif delta_margin >= 0: s2 = 10
            else: s2 = 0
            
            # S3: 成長の質 (Max 20)
            s3 = 0
            if quality_gap >= 0.04: s3 = 20
            elif quality_gap >= 0.01: s3 = 12
            elif quality_gap >= 0: s3 = 6
            else: s3 = 0
            
            self.base_score = s1 + s2 + s3
            # 正規化は省略（主要項目が取れない場合はGatewayで弾かれるためシンプルに合計）

            # --- 5.2 警告フラグ ---
            # F1: 価格要因リスク
            rev_cagr = calc_cagr(df_calc['Revenue'], n_cagr) or 0
            if (delta_margin > 0.02) and (rev_cagr < 0.01):
                self.flags_yellow.append("F1:PriceDriven")
            
            # F3: 構造変化疑い (Assets/Revenueの急変)
            last_assets = df_calc['Capital'].iloc[-1]
            prev_assets_val = df_calc['Capital'].iloc[-2]
            assets_chg = (last_assets - prev_assets_val) / abs(prev_assets_val) if prev_assets_val != 0 else 0
            
            if assets_chg > 0.50 or assets_chg < -0.35:
                self.flags_red.append("F3:StructChange(Assets)")
                
            # F5: Outputマイナス
            if df_calc['Output'].iloc[-1] < 0:
                self.flags_red.append("F5:NegOutput")

            # --- 判定ロジック (Gateway & Overconfidence blocker) ---
            # 仮判定
            temp_grade = "D"
            if self.base_score >= 85: temp_grade = "AA"
            elif self.base_score >= 70: temp_grade = "A"
            elif self.base_score >= 55: temp_grade = "B"
            elif self.base_score >= 40: temp_grade = "C"
            else: temp_grade = "D"

            final = temp_grade
            reasons = []

            # 5.1 ゲート条件
            if self.data_quality_score < 40:
                final = "D"
                reasons.append("Quality<40(Fatal)")
            elif self.data_quality_score < 60:
                if final in ["AA", "A", "B"]:
                    final = "C"
                    reasons.append("Quality<60(Cap:C)")

            # 5.3 強制降格ルール
            # AA制限
            if final == "AA":
                is_ok = True
                if self.data_quality_score < 80: is_ok = False
                if self.output_metric_used not in ['EBITDA', 'Operating Income']: is_ok = False
                if len(self.flags_red) > 0: is_ok = False
                if len(self.flags_yellow) > 2: is_ok = False
                
                if not is_ok:
                    final = "A"
                    reasons.append("AA_Req_Fail")

            # A制限
            if final == "A":
                is_downgrade = False
                if len(self.flags_red) > 0: is_downgrade = True
                if self.data_quality_score < 70: is_downgrade = True
                
                if is_downgrade:
                    final = "B"
                    reasons.append("A_Req_Fail")

            # B制限
            if final == "B":
                if self.data_quality_score < 60: # すでにGateでCだが念のため
                    final = "C"
                if (self.output_metric_used == 'Total Revenue') and (len(self.flags_yellow) > 1):
                    final = "C"
                    reasons.append("B_Req_Fail(RevBase)")

            self.final_grade = final
            self.reason = "; ".join(reasons)
            
            # 数値保存
            self.metrics = {
                'CP_CAGR': f"{cagr_cp:.1%}",
                'DeltaMargin': f"{delta_margin:.1%}",
                'QualityGap': f"{quality_gap:.1%}"
            }

        except Exception as e:
            self.final_grade = "E" # Error
            self.reason = str(e)[:50]

    def get_result_row(self):
        # スプレッドシート出力用の配列を作成
        # C列以降: 判定, BaseScore, Quality, Flags, Reason, Metrics...
        
        y_str = ", ".join(self.flags_yellow)
        r_str = ", ".join(self.flags_red)
        flags_str = f"[Y:{y_str}] [R:{r_str}]" if (y_str or r_str) else "None"
        
        metrics_str = f"CP_CAGR:{self.metrics.get('CP_CAGR','-')}, Gap:{self.metrics.get('QualityGap','-')}"
        
        return [
            self.final_grade,           # C列
            self.base_score,            # D列
            self.data_quality_score,    # E列
            flags_str,                  # F列
            self.reason,                # G列
            self.output_metric_used,    # H列
            metrics_str,                # I列
            datetime.now().strftime('%Y-%m-%d') # J列(更新日)
        ]

# --- メイン処理 ---
def main():
    # 1. GCP認証
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    json_key = json.loads(os.environ['GCP_KEYS_JSON'])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json_key, scope)
    client = gspread.authorize(creds)

    # 2. シートを開く
    sheet_url = os.environ['SPREADSHEET_URL']
    sheet_name = os.environ['SHEET_NAME']
    sheet = client.open_by_url(sheet_url).worksheet(sheet_name)

    # 3. データの読み込み (A列:コード, B列:社名)
    # 全データを取得
    all_values = sheet.get_all_values()
    # ヘッダー行(1行目)を除外するか、そのまま処理するか。ここでは2行目から処理とする
    rows = all_values[1:] 
    
    results = []
    
    print("Starting analysis...")
    
    for i, row in enumerate(rows):
        code_raw = str(row[0]).strip()
        
        # コードが空ならスキップ
        if not code_raw:
            results.append([""] * 8)
            continue
            
        # 日本株コードの正規化 (数字のみ4桁等は .T をつける)
        ticker = code_raw
        if code_raw.isdigit() and len(code_raw) == 4:
            ticker = f"{code_raw}.T"
            
        # 進捗ログ (銘柄名は出さない)
        log_progress(i + 1, len(rows))
        
        # 分析実行
        scorer = TFPScorer(ticker)
        scorer.run_analysis()
        
        results.append(scorer.get_result_row())
        
        # レート制限対策 (重要)
        time.sleep(2.0)

    # 4. 書き込み (C2セル起点)
    # 行数に合わせて範囲を指定
    range_start = "C2"
    range_end = f"J{len(results) + 1}"
    cell_range = f"{range_start}:{range_end}"
    
    print("Updating spreadsheet...")
    sheet.update(cell_range, results)
    print("Done.")

if __name__ == "__main__":
    main()
