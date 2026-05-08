# 📖 论文写作指南 — LNCS 模板

## 目录结构

```
paper/
│
├── main.tex                ← 论文主文件（所有内容写在这里）
├── myReference.bib          ← 参考文献数据库
├── llncs.cls                ← LNCS 文档类（Springer 会议模板，不要修改）
├── splncs04.bst             ← 参考文献样式（不要修改）
│
├── images/                  ← 存放所有图片
│   ├── 1.png, 2.png, ...    ← 主图（网络结构、流程图等）
│   ├── a.png, b.png, ...    ← 对比实验图
│   ├── 4-1.png ~ 7-5.png    ← 子图/详细结果图
│   └── ...
│
├── readme.txt               ← 模板说明
└── history.txt              ← 版本历史
```

## 编译方法

### VS Code 中编译（推荐）

安装 LaTeX Workshop 插件后：

- **一键编译**: `Ctrl+Alt+B`（默认 recipe 是 pdflatex→bibtex→pdflatex→pdflatex）
- **查看 PDF**: `Ctrl+Alt+V`（打开侧边预览）
- **从命令面板**: `Ctrl+Shift+P` → `Build with recipe` → `pdflatex → bibtex → pdflatex → pdflatex`

### 命令行编译

```bash
# 进入论文目录
cd "/home/zah/ORB_SLAM3-master/landmarkslam/implement/paper/An_End_to_end_Approach_for_Chessboard_Corner_Detection_and_Subpixel_Estimation (2)"

# 完整编译链（每次修改后执行）
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## 模板说明

当前使用 **Springer LNCS** 格式（Lecture Notes in Computer Science），是计算机领域最常见的会议模板之一。

### 文档结构（main.tex 中的各部分）

| 行号 | 部分 | 说明 |
|------|------|------|
| 5 | `\documentclass[runningheads]{llncs}` | 文档类声明，不要改动 |
| 7-28 | 导言区 | 调用宏包，可按需添加 |
| 44 | `\title{...}` | 论文标题 |
| 53-65 | `\author{...} \institute{...}` | 作者信息 |
| 90 | `\maketitle` | 生成标题 |
| 92-98 | `abstract` | 摘要 |
| 104-128 | Section 1: Introduction | 引言/贡献 |
| 131-154 | Section 2: Related Work | 相关工作 |
| 157-270 | Section 3: Method | 方法（核心） |
| 273-279 | Section 3.1: Dataset | 数据集 |
| 282-353 | Section 3.2: Loss Function | 损失函数 |
| 360-534 | Section 4: Experiments | 实验与结果 |
| 566-624 | Section 5: Multi-Chessboard | 多棋盘检测 |
| 627-628 | Section 6: Conclusion | 结论 |
| 631-632 | 参考文献 | `\bibliography{myReference}` |

## 常用语法速查

### 图片插入

```latex
% 单张图
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{images/文件名.png}
    \caption{图片标题}
    \label{fig:标签名}
\end{figure}

% 两张图并排
\begin{figure}[htbp]
    \centering
    \begin{minipage}{0.48\textwidth}
        \includegraphics[width=\linewidth]{images/a.png}
        \subcaption{(a) 标题}
    \end{minipage}
    \hfill
    \begin{minipage}{0.48\textwidth}
        \includegraphics[width=\linewidth]{images/b.png}
        \subcaption{(b) 标题}
    \end{minipage}
    \caption{总标题}
    \label{fig:双图}
\end{figure}
```

### 表格

```latex
\begin{table}[htbp]
\centering
\caption{表格标题}
\label{tab:标签}
\begin{tabular}{lccc}
\toprule
\textbf{方法} & \textbf{指标1} & \textbf{指标2} & \textbf{指标3} \\
\midrule
方法A & 90.5 & 85.2 & 0.52 \\
方法B & 95.3 & 92.1 & 0.47 \\
\bottomrule
\end{tabular}
\end{table}
```

### 公式

```latex
% 行内公式
$E = mc^2$

% 独立公式（有编号）
\begin{equation}
    \mathcal{L} = \frac{1}{N} \sum_{i=1}^N \|x_i - \hat{x}_i\|
    \label{eq:loss}
\end{equation}

% 多行公式（无编号）
\begin{equation}
    \begin{aligned}
    a &= b + c \\
    d &= e + f
    \end{aligned}
\end{equation}
```

### 参考文献

```latex
% 在正文中引用
\cite{zhang2002flexible}   → 显示为 [1]
\cite{harris1988,mate2016}  → 显示为 [2,3]

% 新增文献（在 myReference.bib 中添加）
@article{你的标签,
  title={论文标题},
  author={作者},
  journal={期刊名},
  volume={卷},
  number={期},
  pages={页码},
  year={年份}
}
```

### 算法伪代码

```latex
\begin{algorithm}
\caption{算法名称}
\label{alg:标签}
\begin{algorithmic}[1]
\Require 输入
\Ensure 输出
\State 步骤1
\For{$i = 1$ to $N$}
    \State 步骤2
    \If{条件}
        \State 步骤3
    \EndIf
\EndFor
\Return 结果
\end{algorithmic}
\end{algorithm}
```

## 写作建议

1. **修改标题和摘要**: 在 `\title{}` 和 `abstract` 环境中修改
2. **添加章节**: 用 `\section{名称}`、`\subsection{名称}`、`\subsubsection{名称}`
3. **交叉引用**: 用 `\label{...}` 标记，`\ref{...}` 引用（包括图、表、公式、章节）
4. **图片格式**: PNG 优先，也可用 PDF/EPS
5. **每次编译前** 确保 `images/` 中的图片文件存在
6. **参考文献** 统一在 `myReference.bib` 中维护，正文用 `\cite{}` 引用

## 常用宏包（已加载）

| 宏包 | 用途 |
|------|------|
| `graphicx` | 插入图片 |
| `amsmath` | 数学公式 |
| `algorithm` + `algpseudocode` | 算法伪代码 |
| `hyperref` | 超链接（红色 = 引用，蓝色 = URL） |
| `subcaption` | 子图排版 |
| `booktabs` | 专业表格线 |
