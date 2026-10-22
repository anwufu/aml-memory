# AML Memory Service

面向 **CSIG Agent Memory Challenge 2026（第二期）文本赛道 / 开源方法榜** 的
Add / Search 记忆服务。

系统只做两件事：把平台送来的历史写入记忆，按 `user_id` 检索并返回证据。答案生成、
评分、编排全部由平台完成，本服务不生成最终答案。

---

## 1. 设计

```text
Add
  messages -> 逐条切分（长消息按 2000 字符切，带 100 字符重叠）
           -> 每 6 条组装为一个窗口
           -> text-embedding-v4 批量向量化（1024 维，L2 归一化）
           -> PostgreSQL 落库（消息窗口 + 幂等记录 + 会话序号）
           -> 提交后才返回 HTTP 200

Search
  query -> text-embedding-v4 向量化
        -> 稠密召回（该 user_id 下所有窗口，numpy 余弦）
        -> 稀疏召回（该 user_id 下独立统计的 BM25）
        -> RRF 融合
        -> 命中窗口的同会话邻近窗口扩展
        -> 按字符预算裁剪后返回 <= top_k 条证据
```

关键取舍：

- **不使用 pgvector / FAISS。** 每个 `user_id` 只有几十到几百个窗口，按用户取出
  向量做 numpy 矩阵乘法是毫秒级；少一个扩展、少一类部署问题。
- **BM25 的 IDF 按 `user_id` 独立统计**，不使用全局语料统计，避免跨用户共享统计信息。
- **Add 内不调用生成模型。** 保留原始对话，规则/流程类证据不会被摘要压掉
  （这一点对应榜单 G 项，占 31.8% 权重）。
- **幂等键是 `(user_id, request_id)` + payload SHA-256。** 同 ID 同内容返回原响应，
  同 ID 不同内容返回 409。
- **会话序号由 `sessions.next_seq` 原子分配**，保证同一 `session_id` 内窗口顺序稳定，
  邻近扩展可跨 Add 请求工作。

## 2. 契约

| 方法 | 路径 | 鉴权 |
| --- | --- | --- |
| GET | `/health` | 无 |
| POST | `/add`（别名 `/v1/memories/add`） | Bearer / Token / X-Api-Key |
| POST | `/search`（别名 `/v1/memories/search`） | 同上 |

Add 请求：

```json
{
  "request_id": "eval:<run_id>:sample:0:chunk:0",
  "messages": [
    {"role": "user", "content": "raw memory text", "timestamp": 1704067200000}
  ],
  "user_id": "eval:<run_id>:sample:0",
  "session_id": "eval:<run_id>:session:0"
}
```

Add 响应：

```json
{
  "success": true,
  "request_id": "eval:<run_id>:sample:0:chunk:0",
  "user_id": "eval:<run_id>:sample:0",
  "session_id": "eval:<run_id>:session:0"
}
```

Search 请求：

```json
{
  "query": "Which answer best matches the memory?",
  "options": ["A. ...", "B. ..."],
  "user_id": "eval:<run_id>:sample:0",
  "top_k": 100
}
```

Search 响应：

```json
{
  "data": [
    {
      "id": "42",
      "content": "[2024-01-01T00:00:00Z] User: ...\nAssistant: ...",
      "score": 0.0327,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

实现约定：

- `data` 是顶层数组，无 `items` 包装；没有结果返回 `{"data": []}`。
- 返回条数不超过 `top_k`；正式评测 `top_k=100`。
- `score` 是 RRF 融合分，只保证同一次响应内可比。
- `created_at` 来自窗口内最早的消息时间戳。
- 所有 SQL 使用参数占位符；没有任何请求数据拼接进 SQL。

## 3. 本地运行

```bash
cd aml-memory
python -m venv .venv
# Windows
.venv\Scripts\pip install -r requirements.txt
# Linux
# .venv/bin/pip install -r requirements.txt

cp .env.example .env      # 填入数据库、百炼 key、Memory System Key
python scripts/init_db.py # 建表

# Windows
set PYTHONPATH=src && .venv\Scripts\python -m uvicorn aml_memory.api:app --host 127.0.0.1 --port 8000
# Linux
PYTHONPATH=src .venv/bin/python -m uvicorn aml_memory.api:app --host 127.0.0.1 --port 8000

python scripts/smoke_local.py
```

测试：

```bash
python -m pytest -q
```

当前测试覆盖：健康检查、鉴权失败、Add 回传、幂等重放、同 ID 冲突 409、
跨用户隔离、`top_k` 上限、空用户、缺字段 422。

> 测试会在每个用例前执行 `TRUNCATE chunks, add_requests, sessions`。
> 请只对开发库运行测试，不要指向生产库。

## 4. 配置

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | — | PostgreSQL DSN |
| `MEMORY_API_KEY` | — | 平台调用本服务的 Bearer/Token/X-Api-Key |
| `EMBEDDING_API_KEY` | — | 阿里云百炼 key |
| `EMBEDDING_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 兼容模式地址 |
| `EMBEDDING_MODEL` | `text-embedding-v4` | 学术组硬性要求 |
| `EMBEDDING_DIMS` | `1024` | 1024 / 1536 / 2048 |
| `EMBEDDING_BATCH_SIZE` | `10` | 单次请求文本数 |
| `EMBEDDING_CONCURRENCY` | `8` | 并发 embedding 请求数 |
| `WINDOW_MESSAGES` | `6` | 每个证据窗口的消息数 |
| `RRF_K` | `60` | RRF 常数 |
| `NEIGHBOR_SEEDS` | `20` | 做邻近扩展的种子窗口数 |
| `NEIGHBOR_RADIUS` | `1` | 邻近窗口半径（窗口数） |
| `NEIGHBOR_DECAY` | `0.35` | 邻近窗口分数衰减 |
| `TOP_K_MAX` | `100` | 返回上限 |
| `MAX_ITEM_CHARS` | `4000` | 单条证据字符上限 |
| `MAX_TOTAL_CHARS` | `350000` | 整次响应字符预算（约 90k tokens） |

## 5. 部署（阿里云 ECS）

推荐规格：4 vCPU / 8 GB / 100 GB ESSD，Ubuntu 22.04 或 24.04，与百炼同地域。

```bash
# 1. 依赖
sudo apt update && sudo apt install -y python3.12 python3.12-venv postgresql caddy

# 2. 数据库（在服务器上执行一次）
sudo -u postgres psql -c "CREATE USER aml WITH PASSWORD '<长随机密码>';"
sudo -u postgres psql -c "CREATE DATABASE aml_memory OWNER aml;"

# 3. 代码
sudo mkdir -p /opt/aml-memory && sudo chown $USER /opt/aml-memory
git clone <你的仓库> /opt/aml-memory
cd /opt/aml-memory
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env && nano .env
.venv/bin/python scripts/init_db.py
```

systemd（`/etc/systemd/system/aml-memory.service`）：

> 下面的 `User=aml` 指的是 Linux 系统用户，需要先创建（`sudo useradd --system --home /opt/aml-memory aml`）
> 或改成你自己的用户。

```ini
[Unit]
Description=AML Memory Service
After=network.target postgresql.service

[Service]
User=aml
WorkingDirectory=/opt/aml-memory
EnvironmentFile=/opt/aml-memory/.env
Environment=PYTHONPATH=/opt/aml-memory/src
ExecStart=/opt/aml-memory/.venv/bin/python -m uvicorn aml_memory.api:app --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Caddy（`/etc/caddy/Caddyfile`，自动签发 TLS）：

```caddyfile
memory.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now aml-memory
sudo systemctl reload caddy
curl https://memory.example.com/health
```

安全组只开放 22（限制来源 IP）、80、443；**不要对公网开放 5432**。

## 6. 方法与披露

本仓库代码为本次参赛原创实现（MIT）。借鉴的公开方法：

- **Reciprocal Rank Fusion**：Cormack, Clarke, Büttcher, *Reciprocal Rank Fusion
  Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR 2009。
- **混合召回 + 命中窗口邻近扩展**：思路参考 AML 第一期开源文本榜第 1 名
  InvMem（`https://github.com/wenxiaof345-ctrl/vanilla-rag-memory`）。该仓库未附带
  许可证，因此**未复制任何代码**；本实现基于 PostgreSQL + 按用户 BM25 重新设计，
  窗口切分、序号分配、融合和预算裁剪均为独立实现。
- **保留原始对话、不用摘要替代证据**：这是对榜单 G 项（上下文学习与执行，权重约
  31.8%）的针对性设计。相关观察来自 AML 官方第一期榜单分析。

未采用图记忆、原子事实抽取或重排序器；这些方向在公开榜单上表现为 A 项提升但 G 项
下降，整体不划算。

模型合规：

- Embedding：`text-embedding-v4`（1024 维）。
- Add / Search 全链路**不调用生成模型**，因此 `gpt-4o-mini` 项在材料中标注为 N/A。
- 若后续加入查询改写，将只使用 `gpt-4o-mini`，并在材料中更新说明。

## 7. 数据与安全

- 仓库不包含任何评测数据、标准答案、运行产物或密钥。
- `.env` 已被 `.gitignore` 排除；日志不记录请求正文和密钥。
- 评测数据仅用于当前任务，任务完成后 30 天内删除；建议保留删除记录
  （任务 ID、删除时间、执行结果）。
- `user_id` 是唯一检索边界；所有查询都带 `user_id` 过滤，跨用户返回为零。

## 8. 已知限制与下一步

- 离线评测 harness（LoCoMo / LongMemEval / PersonaMem / BEAM / CL-Bench /
  ScriptMem）尚未接入，当前只有契约测试和本地冒烟。
- `1024` vs `2048` 维度的配对 A/B 尚未执行。
- 查询改写、重排序器均未实现。
- 单 worker；Full 规模下的并发、长稳和 embedding 限流需要单独压测。
- Add 事务内会持有数据库连接等待 embedding 返回；已声明 Add 并发 16，连接池上限 20。
