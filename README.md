# KnowFlow

KnowFlow 是一个本地优先的知识助手项目，前端使用 Next.js，后端使用 FastAPI，数据层使用 PostgreSQL + pgvector。项目当前聚焦于多模型对话、知识库 RAG、文档解析、引用溯源、长期记忆和 RAG 检索评估。

> 仓库已增加 `.editorconfig`，统一约定源码使用 UTF-8。若在 Windows PowerShell 中直接查看中文出现乱码，通常是终端输出编码问题，不代表文件内容损坏。

## 功能概览

- AI 对话：支持流式输出、会话持久化、历史会话列表与搜索。
- 多模型配置：前端可配置不同供应商的 API Key、Base URL，并切换模型。
- 知识库 RAG：上传文档后切块、生成向量并写入 pgvector，聊天时可启用知识库检索。
- 引用溯源：RAG 回答会保存并展示引用来源，可跳转或查看原文。
- 文档管理：支持上传、查看、下载、删除文档，并显示处理状态和进度。
- 长期记忆：支持手动添加、语义搜索、删除记忆，并配置自动提取策略。
- 本地 Embedding：可通过 Ollama 使用本地 embedding 模型，减少对云端 embedding API 的依赖。
- RAG 评估：后端提供 Recall@K、MRR、平均延迟等检索评估接口。

## 技术栈

### 前端

- Next.js 16 App Router
- React 19
- TypeScript 5
- Tailwind CSS 4
- Zustand
- shadcn/ui 风格组件
- lucide-react 图标

### 后端

- FastAPI
- SQLAlchemy Async
- PostgreSQL 16
- pgvector
- LiteLLM
- LangChain 相关依赖
- OpenAI SDK
- sentence-transformers
- PyMuPDF / pdfplumber / pypdf / python-docx

## 项目结构

```text
KnowFlow/
├── frontend/                 # Next.js 前端应用
│   ├── app/                  # App Router 页面
│   │   ├── page.tsx          # 首页
│   │   ├── chat/             # 对话页
│   │   ├── knowledge/        # 知识库页
│   │   ├── memories/         # 记忆管理页
│   │   └── settings/         # 模型与 API 配置页
│   ├── components/           # UI 与消息渲染组件
│   ├── lib/                  # API 地址与工具函数
│   ├── stores/               # Zustand 设置存储
│   └── types/                # 前端类型定义
├── backend/                  # FastAPI 后端服务
│   ├── api/                  # 路由：chat/documents/memories/conversations/evaluations
│   ├── core/                 # 配置与数据库连接
│   ├── models/               # SQLAlchemy 模型与 Pydantic Schema
│   └── services/             # LLM、RAG、Embedding、文档、记忆等服务
├── docs/                     # 补充文档
├── screenshots/              # 项目截图
├── uploads/                  # 上传文件目录
├── docker-compose.yml        # PostgreSQL + pgvector
└── start_local_embedding.py  # OpenAI 兼容的本地 embedding 服务示例
```

## 页面截图

![首页](screenshots/homepage.png)

![对话页](screenshots/chat.png)

![知识库](screenshots/knowledge.png)

![记忆管理](screenshots/memories.png)

![设置页](screenshots/settings.png)

## 快速开始

### 1. 使用 Docker Compose 启动完整服务

在仓库根目录执行：

```bash
docker-compose up -d --build
```

该命令会启动：

- `postgres`：PostgreSQL + pgvector，端口 `5432`
- `backend`：FastAPI，端口 `8000`
- `frontend`：Next.js，端口 `3000`

访问地址：

```text
前端：http://localhost:3000
后端：http://localhost:8000
```

如果只需要启动数据库，可执行：

```bash
docker-compose up -d postgres
```

### 2. 本地开发：启动数据库

在仓库根目录执行：

```bash
docker-compose up -d postgres
```

该命令会启动一个 PostgreSQL/pgvector 容器：

- 容器名：`knowflow-db`
- 地址：`localhost:5432`
- 数据库：`knowflow`
- 用户名：`postgres`
- 密码：`postgres`

### 3. 本地开发：启动后端

```bash
cd backend
python -m venv venv
```

Windows PowerShell：

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

macOS / Linux：

```bash
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

后端默认运行在：

```text
http://localhost:8000
```

健康检查：

```text
GET http://localhost:8000/health
```

### 4. 本地开发：启动前端

```bash
cd frontend
npm install
npm run dev
```

前端默认运行在：

```text
http://localhost:3000
```

前端默认通过 `frontend/lib/api.ts` 访问：

```text
http://localhost:8000
```

如果需要修改后端地址，可设置：

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 后端配置

后端配置入口为 `backend/core/config.py`，默认读取 `backend/.env`：

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/knowflow
```

API Key 目前主要通过前端设置页保存在浏览器本地存储中，并随请求发送到后端。设置页支持的供应商包括：

- OpenAI
- Anthropic
- Google AI
- DeepSeek
- 阿里云 DashScope/Qwen
- 智谱 AI
- Moonshot
- Cohere
- Mistral

聊天模型调用经由 LiteLLM。DeepSeek、阿里云、智谱、Moonshot 等 OpenAI 兼容接口会在后端补齐默认 Base URL，也可在前端自定义。

## 知识库与 RAG

知识库页支持上传：

- `.pdf`
- `.docx`
- `.txt`
- `.md`

上传后，后端会：

1. 保存原始文件到 `uploads/`。
2. 解析文本内容。
3. 将文本切块。
4. 生成 embedding。
5. 写入 `document_chunks.embedding` 的 pgvector 字段。
6. 将文档状态更新为 `completed` 或 `failed`。

RAG 检索支持：

- `hybrid`：向量检索 + PostgreSQL 全文检索，并使用 RRF 融合。
- `vector`：仅 pgvector 相似度检索。
- `keyword`：仅 PostgreSQL 全文检索。
- 可选 rerank。
- 可按知识库、文档类型、创建时间、标签和文档 ID 过滤。

更多细节见 [docs/rag_retrieval.md](docs/rag_retrieval.md)。

## 本地 Embedding

前端知识库页会检测本地 Ollama：

```text
http://localhost:11434/api/tags
```

当前代码期望的 Ollama embedding 模型为：

```text
qllama/bge-small-en-v1.5
```

启用本地 embedding 后，文档上传和检索不再要求云端 embedding API Key。后端会调用：

```text
http://localhost:11434/api/embed
```

仓库中还包含 `start_local_embedding.py`，它提供一个 OpenAI 兼容的本地 embedding 服务示例，默认监听 `8088`，但当前主流程使用的是 Ollama 路径。

## 主要接口

### 基础

```text
GET  /
GET  /health
```

### 对话

```text
POST /api/chat
POST /api/chat/rag
GET  /api/tools
```

### 会话

```text
POST   /api/conversations
GET    /api/conversations
GET    /api/conversations/search?q=...
GET    /api/conversations/{conversation_id}
PUT    /api/conversations/{conversation_id}
DELETE /api/conversations/{conversation_id}
POST   /api/conversations/{conversation_id}/messages
```

### 文档

```text
POST   /api/documents/upload
GET    /api/documents/
POST   /api/documents/search
GET    /api/documents/{document_id}
GET    /api/documents/{document_id}/status
GET    /api/documents/{document_id}/content
GET    /api/documents/{document_id}/preview-info
GET    /api/documents/{document_id}/file
DELETE /api/documents/{document_id}
```

### 记忆

```text
POST   /api/memories/
GET    /api/memories/
GET    /api/memories/search
PUT    /api/memories/{memory_id}
DELETE /api/memories/{memory_id}
POST   /api/memories/extract
GET    /api/memories/settings
POST   /api/memories/settings
GET    /api/memories/settings/check-topic
```

### RAG 评估

```text
POST /api/evaluations/rag
```

## 常用命令

### 前端

```bash
cd frontend
npm run dev
npm run build
npm run start
npm run lint
npm run test
```

前端测试使用 Vitest，当前包含 `frontend/lib/utils.test.ts` 作为基础测试用例。

### 后端

```bash
cd backend
python main.py
```

也可使用：

```bash
uvicorn main:app --reload
```

运行后端测试：

```bash
pytest -q
```

当前包含 `backend/tests/test_health.py`，用于验证 `/health` 基础接口。

### 数据库

```bash
docker-compose up -d
docker-compose up -d postgres
docker-compose down
```

## 当前数据模型

核心表包括：

- `documents`：文档元数据、文件路径、状态、知识库名、标签。
- `document_chunks`：文档切块和向量。
- `conversations`：会话标题、模型、消息统计和摘要。
- `messages`：会话消息。
- `citations`：RAG 回答引用的文档切块来源。
- `memories`：长期记忆内容、类别、重要性和 embedding。
- `memory_settings`：记忆自动提取设置。

## 当前状态

- 源码按 UTF-8 维护，并通过 `.editorconfig` 固化编辑器约定。
- 前端已接入 Vitest，并提供 `npm run test`。
- 后端已增加 `tests/` 目录和基础健康检查测试。
- 文档上传支持 PDF、Word、TXT、Markdown，当前前端文件选择也限制在这些格式。
- `docker-compose.yml` 已包含数据库、后端和前端服务；如只需数据库，可单独启动 `postgres` 服务。

## 许可证

当前仓库未看到独立 `LICENSE` 文件。如需开源分发，请先补充许可证文件并在本文档中更新许可信息。
