# WeftMem（纬忆）

面向 **CSIG Agent Memory Challenge 2026（第二期）文本赛道 / 开源方法榜** 的
Add / Search 记忆服务。系统展示名 **WeftMem**，中文名 **纬忆**；代码仓库为
[`anwufu/aml-memory`](https://github.com/anwufu/aml-memory)。

系统只做两件事：把平台送来的历史写入记忆，按 `user_id` 检索并返回证据。答案生成、
评分、编排全部由平台完成，本服务不生成最终答案。

## 官方评测结果

| 项目 | 结果 |
| --- | --- |
| 系统名称 | **WeftMem（纬忆）** |
| 团队 | Team AML |
| 榜单 / 赛道 | 开源方法榜 · 文本赛道 |
| 官方复核成绩 | **57.67** |
| 榜单排名 | **第 42 名** |
| 评测版本 | v0.1.0 |
| 评测提交 commit | [`12e1a94`](https://github.com/anwufu/aml-memory/commit/12e1a94399a6a40ae027893f1ed3c69be4fddc2e) |
| 榜单发布 | 2026-11-19 18:00（北京时间） |

榜单地址：<https://agentmemoryleaderboard.ai/leaderboard>

> 说明：官方复核后的成绩为 **57.67** 分；本 README 的后续更新属于文档更新，
> 不改变已评测的 v0.1.0 代码。评测用公网 ECS 实例已在榜单发布后释放，
> 复现请按第 5 节在本地或自有服务器部署。

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
  （我们离线分析估计这一项对应榜单 G 项，权重约 31.8%；以官方口径为准）。
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
| `EMBEDDING_CONCURRENCY` | `8` | 并发 embedding 请求数（线上设置为 `16`） |
| `WINDOW_MESSAGES` | `6` | 每个证据窗口的消息数 |
| `RRF_K` | `60` | RRF 常数 |
| `NEIGHBOR_SEEDS` | `20` | 做邻近扩展的种子窗口数 |
| `NEIGHBOR_RADIUS` | `1` | 邻近窗口半径（窗口数） |
| `NEIGHBOR_DECAY` | `0.35` | 邻近窗口分数衰减 |
| `TOP_K_MAX` | `100` | 返回上限 |
| `DB_POOL_MAX` | `20` | 数据库连接池上限（线上设置为 `40`） |
| `MAX_ITEM_CHARS` | `4000` | 单条证据字符上限 |
| `MAX_TOTAL_CHARS` | `350000` | 整次响应字符预算（约 90k tokens） |

线上 Full 评测配置：Add / Search 并发 `32 / 32`，embedding 并发 `16`，数据库连接池 `40`。

## 5. 部署（阿里云 ECS）

> 评测用公网 ECS 实例已在 2026-11-19 榜单发布后释放；以下步骤用于本地或自有服务器复现，
> 不再对应一个在线演示地址。

推荐规格：2–4 vCPU / 8 GB / 40–100 GB ESSD，Ubuntu 22.04 或 24.04，与百炼同地域。

```bash
# 1. 依赖
sudo apt update && sudo apt install -y python3.12 python3.12-venv postgresql caddy

# 2. 数据库（在服务器上执行一次）
sudo -u postgres psql -c "CREATE USER aml WITH PASSWORD '<长随机密码>';"
sudo -u postgres psql -c "CREATE DATABASE aml_memory OWNER aml;"

# 3. 代码
sudo mkdir -p /opt/aml-memory && sudo chown $USER /opt/aml-memory
git clone https://github.com/anwufu/aml-memory.git /opt/aml-memory
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

未采用图记忆、原子事实抽取或重排序器；这些方向在我们的离线实验中表现为 A 项提升但
G 项下降，整体不划算。

模型合规（已通过官方复核）：

- Embedding：`text-embedding-v4`（1024 维）。
- Add / Search 全链路**不调用生成模型**，因此 `gpt-4o-mini` 项在材料中标注为 N/A。
- 若后续加入查询改写，将只使用 `gpt-4o-mini`，并在材料中更新说明。

## 7. 数据与安全

- 仓库不包含任何评测数据、标准答案、运行产物或密钥。
- `.env` 已被 `.gitignore` 排除；日志不记录请求正文和密钥。
- 评测数据仅用于当前任务，任务完成后 30 天内删除；建议保留删除记录
  （任务 ID、删除时间、执行结果）。
- 评测用公网 ECS 已释放，实例内评测数据随实例删除；如保留快照，按同一删除计划处理。
- `user_id` 是唯一检索边界；所有查询都带 `user_id` 过滤，跨用户返回为零。

## 8. 工程状态与限制

已完成并上线：

- 离线评测 harness：LoCoMo / LongMemEval / PersonaMem / CL-Bench 等；
- `1024` vs `2048` 维配对 A/B：1024 维在延迟与存储上更优，最终采用 1024；
- 相对时间解析：对时间类查询在证据中追加括号日期，已上线；
- 线上 Full 压测：Add / Search 并发 32 / 32、embedding 并发 16、DB pool 40，
  100 MB 规模通过；
- 评测提交版本：v0.1.0，commit `12e1a94`。

未采用 / 已知限制：

- 未使用图记忆、原子事实抽取或重排序器；查询扩展离线实验为负收益，未上线；
- 单 worker 部署；Add 事务内会持有数据库连接等待 embedding 返回；
- 评测用公网 ECS 已释放，仓库保留可复现代码与部署说明；
- 榜单权重与分项以 AML 官方口径为准。
