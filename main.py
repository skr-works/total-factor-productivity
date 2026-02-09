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

# --- TFP評価クラス (ハイブリッド評価ロジック) ---
class TFPScorer:
    def __init__(self, ticker, idx):
        self.ticker_symbol = ticker
        self.idx = idx
        self.ticker = yf.Ticker(ticker) # Session注入廃止
        
        # 結果保持用
        self.final_grade = "D"
        self.score_annual = 0
        self.score_quarter = 0
        self.trend_status = "-"
        self.data_quality_score = 0
        self.flags_yellow = []
        self.flags_red = []
        self.reason = ""
        self.asof = "-"
        
        # メタデータ
        self.output_metric = "-"
        self.capital_metric = "-"
        
        # 数値データ (表示用)
        self.val_annual_growth = 0.0
        self.val_quarter_growth = 0.0
        self.val_annual_margin = 0.0
        self.val_quarter_margin = 0.0
        self.val_quality_gap = 0.0

    def _calc_cagr(self, series, n):
        if len(series) < n: return None
        try:
            start_val = series.iloc[-(n)]
            end_val = series.iloc[-1]
            if start_val <= 0 or end_val <= 0: return 0
            return (end_val / start_val) ** (1/(n-1)) - 1
        except:
            return 0

    def _scoring_logic(self, growth, margin, quality_gap=None):
        """共通スコアリングロジック (Max 100)"""
        # 1. 成長 (Max 50)
        s1 = 0
        if growth >= 0.06: s1 = 50
        elif growth >= 0.03: s1 = 35
        elif growth >= 0: s1 = 20
        
        # 2. 利益率水準 (Max 30) - 改善ではなく水準で評価
        s2 = 0
        if margin >= 0.10: s2 = 30
        elif margin >= 0.05: s2 = 20
        elif margin > 0: s2 = 10
        
        # 3. 質の加点 (Max 20)
        s3 = 0
        # quality_gapがNoneの場合は成長率で代替判定（簡易）
        q_val = quality_gap if quality_gap is not None else growth
        if q_val > 0: s3 = 20 # 資産増より利益増が大きい、または単に成長している
        
        return s1 + s2 + s3

    def run(self):
        try:
            time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
            
            # --- 1. データ取得 (年次 & 四半期) ---
            try:
                # 年次
                inc_a = self.ticker.income_stmt
                bs_a = self.ticker.balance_sheet
                # 四半期
                inc_q = self.ticker.quarterly_income_stmt
                bs_q = self.ticker.quarterly_balance_sheet
                
                if inc_a.empty or bs_a.empty:
                    self.reason = "No Annual Data"
                    return self._finalize()
                
                # 並べ替え (過去 -> 未来)
                inc_a = inc_a.sort_index(axis=1)
                bs_a = bs_a.sort_index(axis=1)
                if not inc_q.empty: inc_q = inc_q.sort_index(axis=1)
                if not bs_q.empty: bs_q = bs_q.sort_index(axis=1)

                # 最新日付
                if not inc_q.empty:
                    self.asof = str(inc_q.columns[-1].date())
                else:
                    self.asof = str(inc_a.columns[-1].date())

            except Exception as e:
                self.reason = "Download Error"
                safe_log(f"DL Err: {str(e)[:50]}", self.idx)
                return self._finalize()

            # --- 2. 科目マッピング (共通) ---
            def get_series(df, candidates):
                for c in candidates:
                    if c in df.index:
                        return df.loc[c].astype(float), c
                return None, None

            out_candidates = ['EBITDA', 'Operating Income', 'Gross Profit', 'Total Revenue']
            # 年次で科目特定
            out_s_a, out_name = get_series(inc_a, out_candidates)
            self.output_metric = out_name if out_name else "Missing"
            
            # 資本特定 (Total Assets)
            cap_s_a, cap_name = get_series(bs_a, ['Total Assets'])
            self.capital_metric = cap_name if cap_name else "Missing"
            
            # Revenue特定
            rev_s_a, _ = get_series(inc_a, ['Total Revenue', 'Operating Revenue'])

            # --- 3. データ品質スコア ---
            years = len(inc_a.columns)
            q1 = 40 if years >= 5 else (25 if years == 4 else (10 if years == 3 else 0))
            
            q2 = 0
            if out_name in ['EBITDA', 'Operating Income']: q2 = 30
            elif out_name == 'Gross Profit': q2 = 15
            elif out_name == 'Total Revenue': q2 = 0
            
            q3 = 30 if cap_name == 'Total Assets' else 0
            self.data_quality_score = q1 + q2 + q3

            if self.data_quality_score == 0:
                self.reason = "Data Quality 0"
                return self._finalize()

            # ==========================================
            #  A. 年次評価 (Annual Score)
            # ==========================================
            # 計算用DF作成
            df_a = pd.DataFrame({'Output': out_s_a}).dropna()
            
            # 資本平均化
            if cap_s_a is not None:
                prev = cap_s_a.shift(1)
                avg_cap = (cap_s_a + prev) / 2
                df_a = df_a.join(avg_cap.rename('Capital'), how='inner')
            
            if rev_s_a is not None:
                df_a = df_a.join(rev_s_a.rename('Revenue'), how='inner')
            
            df_a = df_a.dropna(subset=['Output', 'Capital'])

            if len(df_a) >= 3:
                # 指標計算
                df_a['CP'] = df_a['Output'] / df_a['Capital']
                df_a['Margin'] = df_a['Output'] / df_a['Revenue'] if 'Revenue' in df_a.columns else 0
                
                # 5年CAGR (なければ3年)
                n = 5 if len(df_a) >= 5 else len(df_a)
                cagr_cp = self._calc_cagr(df_a['CP'], n) or 0
                cagr_out = self._calc_cagr(df_a['Output'], n) or 0
                cagr_cap = self._calc_cagr(df_a['Capital'], n) or 0
                avg_margin = df_a['Margin'].mean()
                
                self.val_annual_growth = cagr_cp
                self.val_annual_margin = avg_margin
                self.val_quality_gap = cagr_out - cagr_cap
                
                self.score_annual = self._scoring_logic(cagr_cp, avg_margin, self.val_quality_gap)
            else:
                self.score_annual = 0 # 計算不能

            # ==========================================
            #  B. 四半期TTM評価 (Quarterly Score)
            # ==========================================
            valid_q = False
            if not inc_q.empty and not bs_q.empty:
                # 科目取得 (年次と同じ科目名を使う)
                out_s_q, _ = get_series(inc_q, [self.output_metric])
                cap_s_q, _ = get_series(bs_q, [self.capital_metric])
                rev_s_q, _ = get_series(inc_q, ['Total Revenue', 'Operating Revenue'])
                
                if out_s_q is not None and cap_s_q is not None:
                    # TTM化 (直近4四半期合計)
                    # 少なくとも8四半期はないと、YoY比較ができない
                    if len(out_s_q) >= 8:
                        # TTM Output: Rolling sum window 4
                        ttm_out = out_s_q.rolling(window=4).sum().dropna()
                        ttm_rev = rev_s_q.rolling(window=4).sum().dropna() if rev_s_q is not None else None
                        
                        # TTM Capital: (Current + 4Q_Ago) / 2  (季節調整)
                        # shift(4) は4つ前のデータ
                        cap_prev_year = cap_s_q.shift(4)
                        ttm_cap = (cap_s_q + cap_prev_year) / 2
                        ttm_cap = ttm_cap.dropna()
                        
                        # 共通インデックスで結合
                        df_q = pd.DataFrame({'Output': ttm_out}).join(ttm_cap.rename('Capital'), how='inner')
                        if ttm_rev is not None:
                            df_q = df_q.join(ttm_rev.rename('Revenue'), how='inner')
                        
                        if len(df_q) >= 2: # 最低でも CurrentとPrior の2点が必要
                            # 指標計算
                            df_q['CP'] = df_q['Output'] / df_q['Capital']
                            df_q['Margin'] = df_q['Output'] / df_q['Revenue'] if 'Revenue' in df_q.columns else 0
                            
                            # 直近TTM vs 1年前TTM (4期前)
                            # df_qはすでにTTM化されている時系列。
                            # indexは四半期末。df_q.iloc[-1]が最新TTM、iloc[-5]が前年同期TTM
                            # しかしdf_qは四半期ごとのスライドデータなので、iloc[-1]とiloc[-5]でYoYになる
                            
                            if len(df_q) >= 5:
                                curr = df_q.iloc[-1]
                                prior = df_q.iloc[-5] # 1年前のTTM
                                
                                # CP Growth (YoY)
                                if prior['CP'] > 0:
                                    growth_q = (curr['CP'] / prior['CP']) - 1
                                else:
                                    growth_q = 0
                                
                                margin_q = curr['Margin']
                                
                                self.val_quarter_growth = growth_q
                                self.val_quarter_margin = margin_q
                                
                                self.score_quarter = self._scoring_logic(growth_q, margin_q, None) # Gapは省略
                                valid_q = True

            # ==========================================
            #  総合判定 & フラグ
            # ==========================================
            
            # トレンド判定
            if valid_q:
                diff = self.score_quarter - self.score_annual
                if diff >= 10: self.trend_status = "改善"
                elif diff <= -10: self.trend_status = "悪化"
                else: self.trend_status = "安定"
            else:
                self.trend_status = "-"

            # フラグ判定 (年次ベース + 最新四半期赤字チェック)
            if not df_a.empty:
                if df_a['Output'].iloc[-1] < 0: self.flags_red.append("Annual_Deficit")
            
            # 四半期直近赤字
            if valid_q and df_q['Output'].iloc[-1] < 0:
                self.flags_red.append("TTM_Deficit")
            
            if self.output_metric == 'Total Revenue':
                self.flags_yellow.append("RevBase")
            
            # ランク判定
            s_a = self.score_annual
            s_q = self.score_quarter
            
            # 基本ロジック
            grade = "D"
            if s_a >= 80 and s_q >= 80: grade = "AA"
            elif s_q >= 80: grade = "A" # 足元絶好調
            elif s_a >= 80 and s_q >= 60: grade = "A" # 基礎盤石
            elif s_a >= 60 and s_q < 50: grade = "B" # 減速懸念
            elif s_q >= 50: grade = "C" # 足元そこそこ
            else: grade = "D"
            
            # ゲート処理
            reasons = []
            if self.data_quality_score < 40:
                grade = "D"
                reasons.append("Quality<40")
            elif self.data_quality_score < 60:
                if grade in ["AA", "A", "B"]:
                    grade = "C"
                    reasons.append("Quality<60")
            
            # フラグ降格
            if len(self.flags_red) > 0:
                if grade in ["AA", "A"]: 
                    grade = "B"
                    reasons.append("RedFlag")
            
            self.final_grade = grade
            self.reason = "; ".join(reasons)

        except Exception as e:
            self.reason = f"Err: {str(e)[:30]}"
            self.final_grade = "D"
            safe_log(f"Process Err: {e}", self.idx)
        
        return self._finalize()

    def _finalize(self):
        # A~B列は呼び出し元で管理。C列以降を返す
        # 列順: C:判定, D:年次Sc, E:四半期Sc, F:トレンド, G:品質, H:フラグ, I:理由, 
        # J:期末, K:Output, L:Capital, M:年次成長, N:四半期成長, O:年次利益率, P:四半期利益率, Q:質
        
        y_str = ",".join(self.flags_yellow)
        r_str = ",".join(self.flags_red)
        flags_disp = f"Y:[{y_str}] R:[{r_str}]" if (y_str or r_str) else "-"

        return [
            self.final_grade,               # C
            self.score_annual,              # D
            self.score_quarter,             # E
            self.trend_status,              # F
            self.data_quality_score,        # G
            flags_disp,                     # H
            self.reason,                    # I
            self.asof,                      # J
            self.output_metric,             # K
            self.capital_metric,            # L
            round(self.val_annual_growth * 100, 2),  # M (%)
            round(self.val_quarter_growth * 100, 2), # N (%)
            round(self.val_annual_margin * 100, 2),  # O (%)
            round(self.val_quarter_margin * 100, 2), # P (%)
            round(self.val_quality_gap * 100, 2)     # Q (pt)
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
                # 空行はスキップ結果を入れる (15列: C~Q)
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
                # エラー時は空の結果を埋める (15列)
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
