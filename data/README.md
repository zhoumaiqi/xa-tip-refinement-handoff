# 审核数据

`original_train_reference/` 保存原始训练清单和原图 SHA256 清单。

`reviewed_20261015/` 保存审核完成的独立 DICOM 副本与分类清单：

| 分类 | 数量 | 清单 | 影像目录 |
| --- | ---: | --- | --- |
| 清晰可见 | 2,944 | `visible.csv` | `images/visible/` |
| 不可见 | 728 | `not_visible.csv` | `images/not_visible/` |
| 不确定（模糊困难数据） | 441 | `uncertain.csv` | `images/uncertain/` |

三类合计 4,113 帧，互不重叠，不含审核备注。下一步增广默认使用 `visible.csv`，另外两类保留作为单独的数据子集。

## GitHub 保存范围

仓库只保存审核分类、清单、校验信息与说明，DICOM 影像和 `images/` 目录仅保留在本地，并通过 `.gitignore` 排除。克隆仓库不会获得影像；在其他机器使用时，需要另行取得影像并重新定位清单中的路径。

Git 忽略规则不是影像备份；强制清理被忽略文件仍可能删除它们，请另做独立备份。

