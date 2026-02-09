import os
import json
import time
import random
import logging
import math
import concurrent.futures
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- 12. 設定パラメータ (定数) ---
WORKERS = 4
BATCH_SIZE = 50
JITTER_MIN = 0.5
JITTER_MAX = 2.0
RETRIES = 3
TIMEOUT = 20

# User-Agent Rotation List
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# --- ロギング設定 (秘匿化) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def safe_log(msg, idx=None):
    """銘柄コードを出さずにログ出力"""
    prefix = f"[Item #{idx}] " if idx is not None else "[System] "
    logger.info(prefix + msg)

# --- 10. HTTP最適化 ---
def get_session():
    """リトライ機能付きのセッションを作成"""
    session = requests.Session()
    retry = Retry(
        total=RETRIES,
        read=RETRIES,
        connect=RETRIES,
        backoff_factor=1, # 指数バックオフ
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # User-Agent設定
    session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
    return session

# --- TFP評価クラス (1銘柄単位のロジック) ---
class TFPScorer:
    def __init__(self, ticker, idx):
        self.ticker_symbol = ticker
        self.idx = idx
        self.ticker = yf.Ticker(ticker, session=get_session())
        
        # 結果保持用
        self.output_metric = "None"
        self.capital_metric = "None"
        self.cp_cagr_n = 0
        self.delta_margin_pt = 0.0
        self.quality_gap_pt = 0.0
        self.base_score = 0
        self.data_quality_score = 0
        self.flags_yellow = []
        self.flags_red = []
        self.final_grade = "D"
        self.reason = ""
        self.asof = "-"

    def _calc_cagr(self, series, n):
        """CAGR計算: (End/Start)^(1/n) - 1"""
        if len(series) < n: return None
        try:
            start_val = series.iloc[-(n)] # n年前 (例: len5なら idx0)
            end_val = series.iloc[-1]     # 最新
            if start_val <= 0 or end_val <= 0: return 0 # 負の値や0は計算不可扱い
            return (end_val / start_val) ** (1/(n-1)) - 1
        except:
            return 0

    def run(self):
        try:
            # 10.3 ジッター (待機)
            time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
            
            # 3. データ取得
            try:
                income = self.ticker.income_stmt
                bs = self.ticker.balance_sheet
                cf = self.ticker.cashflow
                
                # データがない場合
                if income.empty or bs.empty:
                    self.reason = "No Financial Data"
                    return self._finalize()

                # 列を昇順（過去→現在）に並べ替え
                income = income.sort_index(axis=1)
                bs = bs.sort_index(axis=1)
                cf = cf.sort_index(axis=1)
                
                # 日付取得
                self.asof = str(income.columns[-1].date())
                
            except Exception as e:
                self.reason = "Download Error"
                return self._finalize()

            years_count = len(income.columns)

            # --- 8.1.1 Data Quality Score: Q1 (期間) ---
            q1 = 0
            if years_count >= 5: q1 = 40
            elif years_count == 4: q1 = 25
            elif years_count == 3: q1 = 10
            
            # --- 4. 科目マッピング ---
            
            # Helper: 行取得
            def get_series(df, candidates):
                for c in candidates:
                    if c in df.index:
                        return df.loc[c].astype(float), c
                return None, None

            # Output
            out_candidates = ['EBITDA', 'Operating Income', 'Gross Profit', 'Total Revenue']
            output_s, out_name = get_series(income, out_candidates)
            self.output_metric = out_name if out_name else "Missing"

            # Q2 Output Quality
            q2 = 0
            if out_name in ['EBITDA', 'Operating Income']: q2 = 30
            elif out_name == 'Gross Profit': q2 = 15
            elif out_name == 'Total Revenue': q2 = 0
            else: q2 = 0

            # Revenue (For Margin)
            rev_s, _ = get_series(income, ['Total Revenue', 'Operating Revenue'])
            
            # Capital (Assets)
            assets_s, assets_name = get_series(bs, ['Total Assets'])
            ppe_s, ppe_name = get_series(bs, ['Net PPE', 'Property Plant Equipment']) # 参考用
            
            # Capital Logic (Average)
            capital_s = None
            q3 = 0
            capital_name = "Missing"

            if assets_s is not None:
                # 前年との平均
                prev = assets_s.shift(1)
                avg_assets = (assets_s + prev) / 2
                
                # 平均計算可能なデータだけ抽出
                valid_capital = avg_assets.dropna()
                
                if not valid_capital.empty:
                    capital_s = valid_capital
                    capital_name = "AvgAssets"
                    q3 = 30
                else:
                    # 平均できない(1期のみ等) -> 期末
                    capital_s = assets_s
                    capital_name = "AssetsEnd"
                    q3 = 15
            
            self.capital_metric = capital_name
            self.data_quality_score = q1 + q2 + q3

            # --- 計算用データフレーム作成 ---
            # 共通期間で結合
            df_calc = pd.DataFrame({'Output': output_s}).dropna()
            
            if capital_s is not None:
                df_calc = df_calc.join(capital_s.rename('Capital'), how='inner')
            else:
                df_calc['Capital'] = np.nan
            
            if rev_s is not None:
                df_calc = df_calc.join(rev_s.rename('Revenue'), how='inner')
            else:
                df_calc['Revenue'] = np.nan

            df_calc = df_calc.dropna(subset=['Output', 'Capital']) # 必須項目なしは除外

            if len(df_calc) < 2:
                self.reason = "Insufficient calculated periods"
                self.data_quality_score = 0 # 強制的に最低評価
                return self._finalize()

            # 6. 指標定義
            df_calc['CP'] = df_calc['Output'] / df_calc['Capital']
            
            # Margin (Revenueがない場合は計算不可で0埋め)
            if 'Revenue' in df_calc.columns:
                df_calc['Margin'] = df_calc['Output'] / df_calc['Revenue']
            else:
                df_calc['Margin'] = np.nan

            # CAGR計算 (n=5 or 3)
            n_cagr = 5 if len(df_calc) >= 5 else (3 if len(df_calc) >=3 else len(df_calc))
            
            cagr_output = self._calc_cagr(df_calc['Output'], n_cagr) or 0
            cagr_capital = self._calc_cagr(df_calc['Capital'], n_cagr) or 0
            cagr_cp = self._calc_cagr(df_calc['CP'], n_cagr) or 0
            
            self.cp_cagr_n = n_cagr
            self.quality_gap_pt = (cagr_output - cagr_capital) * 100 # %pt

            # Margin Delta (直近3年平均 - 前3年平均)
            margin_delta = 0.0
            margins = df_calc['Margin'].dropna()
            if len(margins) >= 6:
                recent = margins.iloc[-3:].mean()
                past = margins.iloc[-6:-3].mean()
                margin_delta = recent - past
            elif len(margins) >= 4: # 縮退
                recent = margins.iloc[-2:].mean()
                past = margins.iloc[-4:-2].mean()
                margin_delta = recent - past
            
            self.delta_margin_pt = margin_delta * 100 # pt

            # --- 7. スコアリング ---
            # S1: CP Growth (Max 50)
            s1 = 0
            if cagr_cp >= 0.06: s1 = 50
            elif cagr_cp >= 0.03: s1 = 35
            elif cagr_cp >= 0: s1 = 20
            
            # S2: Margin Delta (Max 30)
            s2 = 0
            if self.delta_margin_pt >= 3.0: s2 = 30
            elif self.delta_margin_pt >= 1.0: s2 = 20
            elif self.delta_margin_pt >= 0: s2 = 10

            # S3: Quality Gap (Max 20)
            s3 = 0
            if self.quality_gap_pt >= 4.0: s3 = 20
            elif self.quality_gap_pt >= 1.0: s3 = 12
            elif self.quality_gap_pt >= 0: s3 = 6
            
            # 正規化: 項目欠損時の処理 (簡易的に、取れた項目の比率で100点満点に戻すロジック等は省略し、単純加算とする。ゲートで弾くため)
            self.base_score = s1 + s2 + s3

            # --- 8.2 警告フラグ ---
            # F: Output Minus
            if df_calc['Output'].iloc[-1] < 0:
                self.flags_red.append("NegOutput_Latest")
            if (df_calc['Output'] < 0).sum() >= 2:
                self.flags_red.append("NegOutput_Freq")

            # F: Structural Change (Red)
            if len(df_calc) >= 2:
                last_cap = df_calc['Capital'].iloc[-1]
                prev_cap = df_calc['Capital'].iloc[-2]
                cap_chg = (last_cap - prev_cap) / abs(prev_cap) if prev_cap != 0 else 0
                if cap_chg > 0.50 or cap_chg < -0.35:
                    self.flags_red.append("StructChange_Assets")
                
                if 'Revenue' in df_calc.columns:
                    last_rev = df_calc['Revenue'].iloc[-1]
                    prev_rev = df_calc['Revenue'].iloc[-2]
                    rev_chg = (last_rev - prev_rev) / abs(prev_rev) if prev_rev != 0 else 0
                    if rev_chg > 0.40 or rev_chg < -0.30:
                        self.flags_red.append("StructChange_Rev")

            # F: Price Driven (Yellow)
            if self.output_metric == 'Total Revenue':
                self.flags_yellow.append("RevBase")
            else:
                # 売上横ばい(CAGR<1%)なのにマージン急増(>2pt)
                rev_cagr = 0
                if 'Revenue' in df_calc.columns:
                    rev_cagr = self._calc_cagr(df_calc['Revenue'], n_cagr) or 0
                
                if (rev_cagr < 0.01) and (self.delta_margin_pt > 2.0):
                    self.flags_yellow.append("PriceDriven_Suspicion")

            # F: Cyclical (Yellow) - 簡易判定: CPの符号変化が多い、または標準偏差が高い等
            # ここではシンプルにCPが前年比で大きく振れた回数を見る
            cp_pct_chg = df_calc['CP'].pct_change().dropna()
            if (cp_pct_chg.abs() > 0.3).sum() >= 2: # 30%以上の変動が2回以上
                self.flags_yellow.append("HighVol_Cyclical")

            # CapEx Check (Optional) - データ取得省略のため今回は実装せず (仕様4.4に基づき判定不能)

            # --- 最終判定 (ゲート & 降格) ---
            temp_grade = "D"
            if self.base_score >= 85: temp_grade = "AA"
            elif self.base_score >= 70: temp_grade = "A"
            elif self.base_score >= 55: temp_grade = "B"
            elif self.base_score >= 40: temp_grade = "C"
            else: temp_grade = "D"
            
            final = temp_grade
            reasons = []

            # 8.1.2 ゲート
            if self.data_quality_score < 40:
                final = "D"
                reasons.append("Quality<40(Fatal)")
            elif self.data_quality_score < 60:
                if final in ["AA", "A", "B"]:
                    final = "C"
                    reasons.append("Quality<60(Cap:C)")

            # 8.3 強制降格
            # AA Check
            if final == "AA":
                aa_ok = True
                if self.data_quality_score < 80: aa_ok = False
                if self.output_metric not in ['EBITDA', 'Operating Income']: aa_ok = False
                if len(self.flags_red) > 0: aa_ok = False
                if len(self.flags_yellow) > 2: aa_ok = False
                
                if not aa_ok:
                    final = "A"
                    reasons.append("AA_Req_Fail")

            # A Check
            if final == "A":
                a_downgrade = False
                if len(self.flags_red) > 0: a_downgrade = True
                if self.data_quality_score < 70: a_downgrade = True
                # 構造変化Redがある場合はBへ (上のRedチェックに含まれるが明示)
                
                if a_downgrade:
                    final = "B"
                    reasons.append("A_Req_Fail")

            # B Check
            if final == "B":
                b_downgrade = False
                if self.data_quality_score < 60: b_downgrade = True
                if (self.output_metric == 'Total Revenue') and (len(self.flags_yellow) > 1):
                    b_downgrade = True
                    reasons.append("B_Req_Fail(RevBase+Flags)")
                
                if b_downgrade:
                    final = "C"

            self.final_grade = final
            self.reason = "; ".join(reasons)

        except Exception as e:
            self.reason = f"Err: {str(e)[:30]}"
            self.final_grade = "D"
            safe_log(f"Exception: {e}", self.idx)
        
        return self._finalize()

    def _finalize(self):
        # Google Sheets書き込み用のリストを返す
        # 列順: Ticker, AsOf, FinalGrade, BaseScore, QualityScore, Flags, Reason, 
        # OutputMetric, CapitalMetric, CP_CAGR, DeltaMargin, QualityGap
        
        y_str = ",".join(self.flags_yellow)
        r_str = ",".join(self.flags_red)
        flags_disp = f"Y:[{y_str}] R:[{r_str}]" if (y_str or r_str) else "-"

        return [
            self.final_grade,           # C
            self.base_score,            # D
            self.data_quality_score,    # E
            flags_disp,                 # F
            self.reason,                # G
            self.asof,                  # H
            self.output_metric,         # I
            self.capital_metric,        # J
            f"{self.cp_cagr_n}yr:{self.cp_cagr_n:>.1%}" if self.cp_cagr_n else "-", # K
            round(self.delta_margin_pt, 2), # L
            round(self.quality_gap_pt, 2)   # M
        ]

# --- バッチ処理 ---
def process_batch(batch_data, start_idx):
    """バッチ内の並列処理を実行"""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_idx = {}
        for i, row in enumerate(batch_data):
            # A列: Code, B列: Name (使用しないがoffset用)
            code = str(row[0]).strip()
            if not code:
                # 空行はスキップ結果を入れる
                results.append([""] * 11) 
                continue

            # 日本株コード補正
            ticker = f"{code}.T" if code.isdigit() else code
            
            # 並列タスク投入
            idx = start_idx + i
            scorer = TFPScorer(ticker, idx)
            future = executor.submit(scorer.run)
            future_to_idx[future] = i # バッチ内のインデックスを保持

        # 結果回収（順序維持のためリスト初期化が必要だが、ここではappend後にsortする簡易実装）
        batch_results = [None] * len(batch_data)
        
        for future in concurrent.futures.as_completed(future_to_idx):
            local_i = future_to_idx[future]
            try:
                res = future.result()
                batch_results[local_i] = res
            except Exception as e:
                batch_results[local_i] = ["E", 0, 0, "SysErr", str(e), "-", "-", "-", "-", 0, 0]
        
    return batch_results

# --- メイン ---
def main():
    safe_log("Script started.")
    
    # 1. Config読み込み & GSpread認証
    try:
        # --- 修正箇所: APP_CONFIGから一括読み込み ---
        config_str = os.environ.get('APP_CONFIG')
        if not config_str:
            raise ValueError("APP_CONFIG secret is missing")
        
        config = json.loads(config_str)
        
        # JSONから必要なパラメータを展開
        gcp_key = config['gcp_key']
        sheet_url = config['spreadsheet_url']
        sheet_name = config['sheet_name']

        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(gcp_key, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(sheet_url).worksheet(sheet_name)
        # ------------------------------------------

    except Exception as e:
        safe_log(f"Setup Failed: {e}")
        return

    # 2. データ読み込み
    all_values = sheet.get_all_values()
    # ヘッダー(1行目)除外
    rows = all_values[1:]
    total_rows = len(rows)
    
    safe_log(f"Total rows to process: {total_rows}")

    # 3. バッチループ
    current_idx = 0
    while current_idx < total_rows:
        batch_end = min(current_idx + BATCH_SIZE, total_rows)
        batch_rows = rows[current_idx:batch_end]
        
        safe_log(f"Processing batch {current_idx+1} to {batch_end}...")
        
        # 処理実行
        batch_output = process_batch(batch_rows, current_idx)
        
        # 書き込み (C列〜M列)
        # スプレッドシートの行番号は 2 (header) + current_idx
        start_row = 2 + current_idx
        end_row = start_row + len(batch_output) - 1
        cell_range = f"C{start_row}:M{end_row}"
        
        try:
            sheet.update(cell_range, batch_output)
            safe_log(f"Batch write success: {cell_range}")
        except Exception as e:
            safe_log(f"Batch write failed: {e}")
            # リトライロジックは簡易化のため省略、ログのみ
        
        current_idx += BATCH_SIZE
        time.sleep(1) # バッチ間の安全待機

    safe_log("All done.")

if __name__ == "__main__":
    main()
