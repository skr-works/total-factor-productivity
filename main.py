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
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
    return session

# --- TFP評価クラス (ハイブリッド評価ロジック・厳格版) ---
class TFPScorer:
    def __init__(self, ticker, idx):
        self.ticker_symbol = ticker
        self.idx = idx
        self.ticker = yf.Ticker(ticker)
        
        # 結果保持用
        self.final_grade = "D"
        self.total_score = 0
        self.efficiency_spread = 0.0
        self.quarterly_status = "-"
        self.data_quality_score = 0
        self.flags_yellow = []
        self.flags_red = []
        self.reason = ""
        self.asof = "-"
        
        # メタデータ
        self.output_metric = "-"
        self.capital_metric = "-"
        
        # 数値データ (表示用)
        self.val_out_cagr = 0.0
        self.val_cap_cagr = 0.0
        self.val_margin_trend = "-"
        self.val_consistency_count = 0

        # 内部計算用
        self._cp_cagr = None
        self._delta_margin = None

    def _calc_cagr(self, series, n):
        if series is None: 
            return None
        if n is None or n <= 1:
            return None
        if len(series) < n:
            return None
        try:
            start_val = float(series.iloc[-(n)])
            end_val = float(series.iloc[-1])
            if not (math.isfinite(start_val) and math.isfinite(end_val)):
                return None
            if start_val <= 0 or end_val <= 0:
                return None
            return (end_val / start_val) ** (1/(n-1)) - 1
        except:
            return None

    def _new_scoring_logic(self, spread, last_margin, avg_margin, consistency_count):
        """
        評価基準 (合計100点)
        1. CP_CAGR (50点): Output/Capital の成長
        2. ΔMargin (30点): 利益率の改善（pt）
        3. QualityGap (20点): CAGR(Output) - CAGR(Capital)
        """
        # 1) CP_CAGR (50)
        s1 = 0
        if self._cp_cagr is None:
            s1 = 0
        else:
            if self._cp_cagr >= 0.06: s1 = 50
            elif self._cp_cagr >= 0.03: s1 = 35
            elif self._cp_cagr >= 0.0: s1 = 20
            else: s1 = 0

        # 2) ΔMargin (30)
        s2 = 0
        if self._delta_margin is None:
            s2 = 0
        else:
            if self._delta_margin >= 0.03: s2 = 30
            elif self._delta_margin >= 0.01: s2 = 20
            elif self._delta_margin >= 0.0: s2 = 10
            else: s2 = 0

        # 3) QualityGap (20) = spread
        s3 = 0
        if spread is None:
            s3 = 0
        else:
            if spread >= 0.04: s3 = 20
            elif spread >= 0.01: s3 = 12
            elif spread >= 0.0: s3 = 6
            else: s3 = 0

        return s1 + s2 + s3

    def run(self):
        try:
            time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
            
            # --- 1. データ取得 ---
            try:
                inc_a = self.ticker.income_stmt
                bs_a = self.ticker.balance_sheet
                inc_q = self.ticker.quarterly_income_stmt # 確認用
                
                if inc_a.empty or bs_a.empty:
                    self.reason = "No Annual Data"
                    return self._finalize()
                
                # 並べ替え
                inc_a = inc_a.sort_index(axis=1)
                bs_a = bs_a.sort_index(axis=1)
                if not inc_q.empty: inc_q = inc_q.sort_index(axis=1)

                self.asof = str(inc_a.columns[-1].date())

            except Exception as e:
                self.reason = "Download Error"
                safe_log(f"DL Err: {type(e).__name__}", self.idx)
                return self._finalize()

            # --- 2. 科目マッピング ---
            def get_series(df, candidates):
                for c in candidates:
                    if c in df.index:
                        return df.loc[c].astype(float), c
                return None, None

            out_candidates = ['EBITDA', 'Operating Income', 'Gross Profit', 'Total Revenue']
            out_s_a, out_name = get_series(inc_a, out_candidates)
            self.output_metric = out_name if out_name else "Missing"
            
            cap_s_a, cap_name = get_series(bs_a, ['Total Assets'])
            self.capital_metric = cap_name if cap_name else "Missing"
            
            rev_s_a, _ = get_series(inc_a, ['Total Revenue', 'Operating Revenue'])

            # Margin用（Outputとは独立）
            op_s_a, _ = get_series(inc_a, ['Operating Income'])
            ebitda_s_a, _ = get_series(inc_a, ['EBITDA'])

            # 必須メトリクス欠損ガード
            if out_s_a is None or cap_s_a is None:
                self.reason = "Missing Required Metric"
                return self._finalize()

            # --- 3. データ品質スコア ---
            years = len(inc_a.columns)
            q1 = 40 if years >= 5 else (25 if years == 4 else (10 if years == 3 else 0))
            
            q2 = 0
            if out_name in ['EBITDA', 'Operating Income']: q2 = 30
            elif out_name == 'Gross Profit': q2 = 15
            elif out_name == 'Total Revenue': q2 = 0
            
            q3 = 30 if cap_name == 'Total Assets' else 0
            self.data_quality_score = q1 + q2 + q3

            if self.data_quality_score < 40:
                self.reason = "Quality<40"
                return self._finalize()

            # ==========================================
            #  本番計算 (年次ベース)
            # ==========================================
            df = pd.DataFrame({'Output': out_s_a}).dropna()
            
            prev = cap_s_a.shift(1)
            avg_cap = (cap_s_a + prev) / 2
            df = df.join(avg_cap.rename('Capital'), how='inner')
            
            if rev_s_a is not None:
                df = df.join(rev_s_a.rename('Revenue'), how='inner')

            if op_s_a is not None:
                df = df.join(op_s_a.rename('OpIncome'), how='left')
            elif ebitda_s_a is not None:
                df = df.join(ebitda_s_a.rename('OpIncome'), how='left')
            
            df = df.dropna(subset=['Output', 'Capital'])

            if len(df) < 3:
                self.reason = "Insufficient periods"
                return self._finalize()

            # Margin計算（Output/Revenueは禁止。Operating Income(or EBITDA)/Revenueに固定）
            if 'Revenue' in df.columns and 'OpIncome' in df.columns:
                df['Margin'] = df['OpIncome'] / df['Revenue']
            else:
                df['Margin'] = np.nan

            # CAGR計算
            n = 5 if len(df) >= 5 else len(df)

            cagr_out = self._calc_cagr(df['Output'], n)
            cagr_cap = self._calc_cagr(df['Capital'], n)

            # CP_CAGR（Output/Capital）
            cp_series = (df['Output'] / df['Capital']).replace([np.inf, -np.inf], np.nan).dropna()
            self._cp_cagr = self._calc_cagr(cp_series, min(n, len(cp_series))) if len(cp_series) >= 3 else None

            # QualityGap = CAGR(Output) - CAGR(Capital)
            spread = None
            if cagr_out is not None and cagr_cap is not None:
                spread = cagr_out - cagr_cap
            self.efficiency_spread = spread if spread is not None else 0.0

            # ΔMargin（直近 vs その前。足りない場合は縮退）
            margins = df['Margin'].dropna()
            self._delta_margin = None
            if len(margins) >= 6:
                recent = margins.iloc[-3:].mean()
                prev_m = margins.iloc[-6:-3].mean()
                self._delta_margin = (recent - prev_m)
            elif len(margins) >= 4:
                recent = margins.iloc[-2:].mean()
                prev_m = margins.iloc[-4:-2].mean()
                self._delta_margin = (recent - prev_m)
            elif len(margins) >= 2:
                self._delta_margin = (margins.iloc[-1] - margins.iloc[-2])

            # 利益率トレンド表示
            if self._delta_margin is None:
                self.val_margin_trend = "-"
                last_margin = 0.0
                avg_margin = 0.0
            else:
                if self._delta_margin >= 0.005: self.val_margin_trend = "改善"
                elif self._delta_margin <= -0.005: self.val_margin_trend = "悪化"
                else: self.val_margin_trend = "維持"
                last_margin = float(margins.iloc[-1]) if len(margins) >= 1 else 0.0
                avg_margin = float(margins.mean()) if len(margins) >= 1 else 0.0

            # Consistency（表示用：増益回数）
            pct_chg = df['Output'].pct_change().dropna()
            consistency = (pct_chg >= 0).sum()
            self.val_consistency_count = int(consistency)
            
            # スコア算出（仕様準拠）
            self.total_score = self._new_scoring_logic(spread, last_margin, avg_margin, consistency)
            
            # --- 直近四半期の確認 (参考情報・フラグ用) ---
            has_quarterly = False
            q_status = "-"
            
            if not inc_q.empty:
                q_out, _ = get_series(inc_q, [self.output_metric])
                if q_out is not None:
                    q_out = q_out.dropna()
                    if len(q_out) >= 1:
                        last_q_val = q_out.iloc[-1]
                        if last_q_val < 0: q_status = "赤字"
                        elif len(q_out) >= 5: # YoY
                            prior_q = q_out.iloc[-5]
                            if prior_q > 0 and last_q_val > prior_q: q_status = "増益"
                            elif prior_q > 0: q_status = "減益"
                            else: q_status = "黒字"
                        else:
                            q_status = "データ有"
                        has_quarterly = True
            
            self.quarterly_status = q_status
            if not has_quarterly:
                self.flags_yellow.append("Annual_Only")

            # --- 警告フラグ ---
            if df['Output'].iloc[-1] < 0: self.flags_red.append("Deficit")
            if self.output_metric == 'Total Revenue': self.flags_red.append("RevBase")

            # 構造変化（資産・売上の急変）
            try:
                assets_end = cap_s_a.dropna()
                if len(assets_end) >= 2:
                    a_prev = float(assets_end.iloc[-2])
                    a_now = float(assets_end.iloc[-1])
                    if a_prev > 0:
                        a_yoy = (a_now / a_prev) - 1
                        if a_yoy > 0.50 or a_yoy < -0.35:
                            self.flags_red.append("Struct_Assets")
                if rev_s_a is not None:
                    rev_end = rev_s_a.dropna()
                    if len(rev_end) >= 2:
                        r_prev = float(rev_end.iloc[-2])
                        r_now = float(rev_end.iloc[-1])
                        if r_prev > 0:
                            r_yoy = (r_now / r_prev) - 1
                            if r_yoy > 0.40 or r_yoy < -0.30:
                                self.flags_red.append("Struct_Revenue")
            except:
                pass
            
            # ==========================================
            #  総合判定 (グレーディング・キャップ制)
            # ==========================================
            grade = "D"
            reasons = []
            
            # 1. ベース判定（AA/A/B/C/D）
            score = self.total_score
            if score >= 85: grade = "AA"
            elif score >= 70: grade = "A"
            elif score >= 55: grade = "B"
            elif score >= 40: grade = "C"
            else: grade = "D"
            
            # 2. 品質ゲート
            if self.data_quality_score < 40:
                grade = "D"
                reasons.append("Quality<40")
            elif self.data_quality_score < 60:
                if grade in ["AA", "A", "B"]:
                    grade = "C"
                    reasons.append("Cap:Quality<60")

            # [Cap: B] Revenueベースは高評価禁止（AA/A不可）
            if self.output_metric == 'Total Revenue':
                if grade in ["AA", "A"]:
                    grade = "B"
                    reasons.append("Cap:RevBase")

            # [Cap: A] Annual OnlyはAA不可
            if not has_quarterly:
                if grade == "AA":
                    grade = "A"
                    reasons.append("Cap:AnnualOnly")

            # 3. Red Flagの扱い（DeficitはD固定、それ以外は上限B）
            if "Deficit" in self.flags_red:
                grade = "D"
                reasons.append("RedFlag:Deficit")
            elif len(self.flags_red) > 0:
                if grade in ["AA", "A"]:
                    grade = "B"
                    reasons.append("Cap:RedFlag")
            
            self.final_grade = grade
            self.reason = "; ".join(reasons)
            
            # 数値保存（表示用：Noneは0に寄せる）
            self.val_out_cagr = cagr_out if cagr_out is not None else 0.0
            self.val_cap_cagr = cagr_cap if cagr_cap is not None else 0.0

        except Exception as e:
            self.reason = f"Err: {type(e).__name__}"
            self.final_grade = "D"
            safe_log(f"Process Err: {type(e).__name__}", self.idx)
        
        return self._finalize()

    def _finalize(self):
        # 列定義: C:判定, D:総合Sc, E:スプレッド, F:四半期状況, G:品質, H:フラグ, I:理由, 
        # J:期末, K:Output, L:Capital, M:Output成長, N:Capital成長, O:利益率トレンド, P:安定回数, Q:(空き)
        
        y_str = ",".join(self.flags_yellow)
        r_str = ",".join(self.flags_red)
        flags_disp = f"Y:[{y_str}] R:[{r_str}]" if (y_str or r_str) else "-"

        return [
            self.final_grade,               # C
            self.total_score,               # D
            round(self.efficiency_spread * 100, 2), # E (%)
            self.quarterly_status,          # F
            self.data_quality_score,        # G
            flags_disp,                     # H
            self.reason,                    # I
            self.asof,                      # J
            self.output_metric,             # K
            self.capital_metric,            # L
            round(self.val_out_cagr * 100, 2), # M (%)
            round(self.val_cap_cagr * 100, 2), # N (%)
            self.val_margin_trend,          # O
            self.val_consistency_count,     # P
            "-"                             # Q (予備)
        ]

# --- バッチ処理 ---
def process_batch(batch_data, start_idx):
    """バッチ内の並列処理を実行"""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_idx = {}
        for i, row in enumerate(batch_data):
            code = str(row[0]).strip()
            if not code:
                results.append([""] * 15) 
                continue

            ticker = f"{code}.T" if code.isdigit() else code
            idx = start_idx + i
            scorer = TFPScorer(ticker, idx)
            future = executor.submit(scorer.run)
            future_to_idx[future] = i

        batch_results = [None] * len(batch_data)
        for future in concurrent.futures.as_completed(future_to_idx):
            local_i = future_to_idx[future]
            try:
                batch_results[local_i] = future.result()
            except:
                batch_results[local_i] = ["E"] + ["-"] * 14
        
    return batch_results

# --- メイン ---
def main():
    safe_log("Script started.")
    
    # 1. Config読み込み & GSpread認証
    try:
        config_str = os.environ.get('APP_CONFIG')
        if not config_str:
            raise ValueError("APP_CONFIG secret is missing")
        
        config = json.loads(config_str)
        gcp_key = config['gcp_key']
        sheet_url = config['spreadsheet_url']
        sheet_name = config['sheet_name']

        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(gcp_key, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(sheet_url).worksheet(sheet_name)

    except Exception as e:
        safe_log(f"Setup Failed: {type(e).__name__}")
        return

    # 2. データ読み込み
    all_values = sheet.get_all_values()
    rows = all_values[1:]
    total_rows = len(rows)
    
    safe_log(f"Total rows to process: {total_rows}")

    # 3. バッチループ
    current_idx = 0
    while current_idx < total_rows:
        batch_end = min(current_idx + BATCH_SIZE, total_rows)
        batch_rows = rows[current_idx:batch_end]
        
        safe_log(f"Processing batch {current_idx+1} to {batch_end}...")
        
        batch_output = process_batch(batch_rows, current_idx)
        
        # 書き込み (C列〜Q列)
        start_row = 2 + current_idx
        end_row = start_row + len(batch_output) - 1
        cell_range = f"C{start_row}:Q{end_row}"
        
        try:
            sheet.update(cell_range, batch_output)
            safe_log(f"Batch write success: {cell_range}")
        except Exception as e:
            safe_log(f"Batch write failed: {type(e).__name__}")
        
        current_idx += BATCH_SIZE
        time.sleep(1)

    safe_log("All done.")

if __name__ == "__main__":
    main()
