Windows 专用依赖与路径检查报告

概述
- 我在仓库中发现若干硬编码的 Windows 路径和 Windows 专用编译/环境检测逻辑（例如 MSVC / LaunchDevCmd）。
- 下面列出发现的位置、具体字符串/行为，以及针对 Linux 迁移的建议动作。

发现项

1. 脚本：scripts/train_ar.py
- 位置/片段：`ensure_msvc_cl_on_path()`、`bootstrap_vs_build_env()` 等函数（见约 300–470 行与 1080–1110 行）。
- 具体行为：自动查找 `cl.exe`、读取 `ProgramFiles`/`ProgramFiles(x86)` 环境变量、调用 `LaunchDevCmd.bat`、拼接 `C:\Program Files (x86)\Windows Kits\10` 等路径；并以此决定是否启用 `torch.compile`。
- 风险/影响：仅在 Windows 有意义；在 Linux 上应跳过或使用等价的 GCC/Clang 检测逻辑；硬编码的 `LaunchDevCmd.bat` 路径可能因 VS 版本不同而失败。
- 建议：保持这段逻辑但确保严格由 `os.name == 'nt'` 保护（代码已如此）；将硬编码路径替换为可配置项或使用 `vswhere` 查找 VS 安装；为 Linux 增加检测 `gcc`/`clang` 分支（如需要支持本地编译优化）。

2. 脚本：tools/data_prep/export_inkml_hdf5.py
- 位置/片段：文件顶部 docstring 与 `argparse` 默认值（默认 `--output_dir` 为 `C:\Projects\LatexProject\datasets`）。
- 具体字符串：`输出目录默认：C:\Projects\LatexProject\datasets`、`parser.add_argument(..., default=r"C:\Projects\LatexProject\datasets")`。
- 风险/影响：在 Linux 上这些默认路径不存在，会导致目录混乱或需要手动传参。
- 建议：将默认改为 `os.path.join(PROJECT_ROOT, "datasets")` 或 `default=os.environ.get('DATASETS_DIR', os.path.join(PROJECT_ROOT, 'datasets'))`，并在 docstring/帮助里说明可通过 CLI 或环境变量覆盖。

3. 脚本：scripts/eval.py
- 位置/片段：命令行参数默认值（约 602–603 行）。
- 具体字符串：`--eval_h5` 与 `--tokenizer` 的默认值使用 Windows 路径（例如 `C:\Projects\LatexProject\ConvResFormula\datasets\val.h5`）。
- 建议：改用基于 `PROJECT_ROOT` 的默认或从配置文件读取（如 `configs/train_ar.yaml` 中的路径）。

4. 模块：src/data/bpe_tokenizer.py
- 位置/片段：多个硬编码的 `WIKI_PARQUET` / 2D datasets 路径（约行 176–181）。
- 具体字符串：例如 `r"C:\Projects\LatexProject\datasets\wikipedia-latex-formulas-319k.parquet"`。
- 建议：改为相对路径或从配置/环境变量读取。把测试/开发资源路径移到 `configs/` 或 `tools/` 的参数里。

5. 多个工具脚本（tools/data_prep/render_inkml_to_image.py、convert_dynamic_resolution.py、auto_build_2d_datasets.py 等）
- 位置/片段：这些脚本在 CLI 默认参数或注释/示例中包含 Windows 路径（如 `C:\Projects\LatexProject\datasets\synthetic.h5` 等）。
- 建议：统一把默认路径替换为相对/项目内路径（基于 `PROJECT_ROOT`），并把示例改成使用 `${PROJECT_ROOT}` 或说明 `--output_dir` 必须设置。

总建议（迁移到 Linux）
- 替换硬编码的 Windows 根路径为：
  - `os.path.join(PROJECT_ROOT, "datasets")`（或 `Path(PROJECT_ROOT)/"datasets"`），并在 CLI 默认中使用该表达式；或
  - 从环境变量读取（例如 `DATASETS_DIR`、`TOKENIZER_PATH`），在缺失时回退到项目内相对路径。
- 对于 MSVC / VS 专有逻辑：保留以支持 Windows 开发者，但不要在程序启动时触发 Linux 分支；为 Linux 加入检测 `gcc`/`clang` 的分支（如果需要本地编译/优化）。
- 在项目根或 `configs/` 中集中管理常用路径：把 `tokenizer_bpe.json`、`datasets` 等路径放入 `configs/train_ar.yaml` 或 `configs/paths.yaml`，脚本从配置加载。
- 将文档与示例（README、脚本帮助）更新为跨平台示例，展示 Linux 下的默认/建议路径与命令行覆盖方法。

下一步我可以：
- 1) 自动把这些硬编码默认替换为基于 `PROJECT_ROOT` 的默认值并提交补丁；或
- 2) 仅生成一份补丁建议（diff）供你审核；或
- 3) 在 `configs/` 中添加 `paths.yaml` 并修改脚本以从中读取路径。

请选择你想要的下一步（例如：`1` 执行自动替换，`2` 只生成补丁建议，或 `3` 创建 config 并修改脚本）。
