# 0. セットアップ（リポジトリを clone して GUI を動かすまで）

`.gitignore` により `data/raw/`（取得データ）・`data/cache/`（流域モデル・スピンアップ等）・`outputs/`・`.venv/` は
リポジトリに含まれない。これらはすべて**公開データから自動で取得・再計算される**ので、手作業でデータを置く必要はない。
必要なのは「環境構築」と「流域ごとの事前計算（prewarm）」だけである。

## 必要なもの

- Python 3.11 以上（動作確認: 3.14.6, numba 0.68）
- インターネット接続（下記サイトから取得）
- ディスク 約1 GB（4流域すべての場合。玖珠川のみなら約100 MB）
- CPUコア数が多いほど速い（全域モデルは Numba 並列）

| 取得元 | 内容 | 保存先 |
|---|---|---|
| cyberjapandata.gsi.go.jp | 標高タイル（DEM10B, DEM5A） | `data/raw/gsi/` |
| nlftp.mlit.go.jp | 国土数値情報（土地利用細分メッシュ, 河川W05, 医療・学校・公共施設） | `data/raw/ksj/` |
| www.e-stat.go.jp | 国勢調査・経済センサスのメッシュ統計 | `data/raw/estat/` |
| www1.river.go.jp | 水文水質DB（観測流量・水位・ダム） | `data/raw/mlit_water/` |
| www.data.jma.go.jp | アメダス時間降水量 | `data/raw/jma/` |

雨量観測所データの格子化に使う中間ファイル（`data/processed/rain_gauges_*.pkl`）はリポジトリに含まれているため、雨量の再取得は不要。

## 手順

### 1. clone と環境構築（約1分）

```bash
git clone https://github.com/tnakash/rri-python.git
cd rri-python
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### 2. 動作確認（任意）

```bash
.venv/bin/python -m pytest
```

### 3. 流域ごとの事前計算（prewarm）

GUI で使う流域・洪水イベントごとに、データ取得 → 流域モデル構築 → 資産・指標データ → スピンアップ → 現況計算 を行い、
`data/cache/` に保存する。使う流域だけ実行すればよい。

```bash
.venv/bin/python scripts/prewarm_gui.py kusu_ogase     kusu_2020             # 玖珠川            約5分
.venv/bin/python scripts/prewarm_gui.py abukuma_akutsu abukuma_akutsu_2019   # 阿武隈川上流      約10分
.venv/bin/python scripts/prewarm_gui.py chikugo_whole  chikugo_2020          # 筑後川全域        約20分
.venv/bin/python scripts/prewarm_gui.py abukuma_whole  abukuma_2019          # 阿武隈川全域      約35分
```

- 時間は12コアのMacでの目安。大半はデータのダウンロード（特に全域の DEM5A タイル）とスピンアップ計算。
- 玖珠川の他のイベント（`kusu_2018`, `kusu_2012`, `kusu_2021`）は、同じ流域モデルを使い回すため数分で済む。
- 成功すると最後に次のような行が出る（玖珠川の例）。値がこれと一致すれば正しく構築されている。
  ```
  session 92s: cells 8448, households 10124, population 25783, ...
  baseline 141s (sim 29s): damage 305 億円, inundated 17.6 km2, livability 52.1, ecology 69.0, ..., mass err -1.0e-14
  ```
- prewarm を省略しても GUI は動くが、初めてその流域を選んだときに同じ処理が GUI 内で走り、長時間待つことになる。

### 4. GUI の起動

```bash
.venv/bin/python -m rri_sim.gui          # http://127.0.0.1:8765 がブラウザで開く
```

使い方は [05_gui.md](05_gui.md) を参照。prewarm 後の施策計算は1ケースあたり 玖珠川 約20秒、筑後川全域 約2分、阿武隈川全域 約5分。

## 計算を省略したい場合

手元で prewarm 済みの人から `data/cache/`（4流域で約120 MB）をコピーしてもらえば、手順3を省略できる
（コードのバージョンが同じであること）。`data/raw/` は不要。

## トラブルシューティング

- **ダウンロードの途中で失敗した**: そのまま同じコマンドを再実行する。取得済みのファイルは再利用される。
  壊れたファイルが残って失敗し続ける場合は、エラーに出たファイル（`data/raw/` 以下）を削除して再実行する。
- **コードを更新した後に結果がおかしい**: `data/cache/` 内の該当流域のファイル（`domain_*`, `assets_*`, `indicator_data_*`, `spinup_*`）を削除して prewarm し直す。
  `data/raw/` は消さなくてよい。
- **`data/raw/` を消してしまった**: `data/cache/` が残っていれば GUI はそのまま動く。モデルを作り直すときに自動で再取得される。
- **GUI のブラウザテスト**（`scripts/gui_smoke_test.py`, `gui_climate_test.py`）を使う場合のみ、追加で `.venv/bin/playwright install chromium` が必要。
