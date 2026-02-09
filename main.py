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

    def _calc_cagr(self, series, n):
        if len(series) < n: return None
        try:
            start_val = series.iloc[-(n)]
            end_val = series.iloc[-1]
            if start_val <= 0 or end_val <= 0: return 0
            return (end_val / start_val) ** (1/(n-1)) - 1
        except:
            return 0

    def _new_scoring_logic(self, spread, last_margin, avg_margin, consistency_count):
        """
        新・評価基準 (合計100点) - 厳格モード
        """
        # 1. Efficiency Spread (50点): 基準引き上げ
        s1 = 0
        if spread >= 0.08: s1 = 50      # +8%以上 (超高効率)
        elif spread >= 0.04: s1 = 30    # +4%以上 (優良)
        elif spread > 0: s1 = 10        # プラス圏 (維持)
        else: s1 = 0                    # マイナス (資産膨張)
        
        # 2. Margin Power (30点): 絶対水準重視
        s2_base = 0
        if last_margin >= 0.15: s2_base = 15    # 15%以上
        elif last_margin >= 0.08: s2_base = 8   # 8%以上
        elif last_margin > 0: s2_base = 2       # 黒字
        
        s2_improv = 0
        if last_margin >= avg_margin + 0.01: s2_improv = 15 # 1pt改善
        elif last_margin >= avg_margin: s2_improv = 5 # 維持
        else: s2_improv = 0
        
        s2 = s2_base + s2_improv
        
        # 3. Consistency (20点)
        s3 = 0
        if consistency_count >= 4: s3 = 20
        elif consistency_count >= 3: s3 = 10
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
                safe_log(f"DL Err: {str(e)[:50]}", self.idx)
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

            # --- 3. データ品質スコア ---
            years = len(inc_a.columns)
            q1 = 40 if years >= 5 else (25 if years == 4 else 0)
            
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
            
            if cap_s_a is not None:
                prev = cap_s_a.shift(1)
                avg_cap = (cap_s_a + prev) / 2
                df = df.join(avg_cap.rename('Capital'), how='inner')
            
            if rev_s_a is not None:
                df = df.join(rev_s_a.rename('Revenue'), how='inner')
            
            df = df.dropna(subset=['Output', 'Capital'])

            if len(df) < 3:
                self.reason = "Insufficient periods"
                return self._finalize()

            # 指標計算
            df['Margin'] = df['Output'] / df['Revenue'] if 'Revenue' in df.columns else 0
            
            # CAGR計算
            n = 5 if len(df) >= 5 else len(df)
            cagr_out = self._calc_cagr(df['Output'], n) or 0
            cagr_cap = self._calc_cagr(df['Capital'], n) or 0
            
            # 指標1: Efficiency Spread (最重要)
            spread = cagr_out - cagr_cap
            self.efficiency_spread = spread
            
            # 指標2: Margin Power
            last_margin = df['Margin'].iloc[-1]
            avg_margin = df['Margin'].mean()
            if last_margin >= avg_margin + 0.005: self.val_margin_trend = "改善"
            elif last_margin <= avg_margin - 0.005: self.val_margin_trend = "悪化"
            else: self.val_margin_trend = "維持"
            
            # 指標3: Consistency
            pct_chg = df['Output'].pct_change().dropna()
            consistency = (pct_chg >= 0).sum()
            self.val_consistency_count = consistency
            
            # スコア算出
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
            
            # ==========================================
            #  総合判定 (グレーディング・キャップ制)
            # ==========================================
            grade = "D"
            reasons = []
            
            # 1. ベース判定
            score = self.total_score
            if score >= 85: grade = "AA"
            elif score >= 70: grade = "A"
            elif score >= 50: grade = "B"
            else: grade = "C"
            
            # 2. 強制キャップ (上限規制)
            
            # [Cap: B] Revenueベースは信用しない
            if self.output_metric == 'Total Revenue':
                if grade in ["AA", "A"]:
                    grade = "B"
                    reasons.append("Cap:RevBase")
            
            # [Cap: B] 資産効率が悪化しているなら高評価しない
            if spread <= 0:
                if grade in ["AA", "A"]:
                    grade = "B"
                    reasons.append("Cap:BadSpread")
            
            # [Cap: A] Annual OnlyはAA不可
            if not has_quarterly:
                if grade == "AA":
                    grade = "A"
                    reasons.append("Cap:AnnualOnly")

            # 3. 降格 (Red Flag)
            if len(self.flags_red) > 0:
                grade = "D" # 赤字等はDまで落とす
                reasons.append("RedFlag")

            self.final_grade = grade
            self.reason = "; ".join(reasons)
            
            # 数値保存
            self.val_out_cagr = cagr_out
            self.val_cap_cagr = cagr_cap
            self.val_annual_margin = avg_margin

        except Exception as e:
            self.reason = f"Err: {str(e)[:30]}"
            self.final_grade = "D"
            safe_log(f"Process Err: {e}", self.idx)
        
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
        safe_log(f"Setup Failed: {e}")
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
            safe_log(f"Batch write failed: {e}")
        
        current_idx += BATCH_SIZE
        time.sleep(1)

    safe_log("All done.")

if __name__ == "__main__":
    main()
