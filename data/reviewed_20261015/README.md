# 审核完成数据集

`all_reviewed.csv` 保存全部 4,113 帧；`visible.csv`、`not_visible.csv`、`uncertain.csv` 分别保存 2,944、728、441 帧。

`images/` 是实际 DICOM 副本，不做增广、裁剪或重编码。GT 和 Raw 坐标保持原始像素坐标，不包含备注，也不包含验证集、测试集。

`checksums.csv` 为之前保存的逐帧 SHA256 清单；本次副本已逐张按此清单校验。`series_mapping.csv` 对应短目录名与 Series UID。

清单包含原始路径、副本绝对路径和包内相对路径。移动数据包后使用相对路径重新定位，不要直接沿用旧绝对路径。

恢复过程说明见 `summary.json`。这是原有人工审核结果的恢复，不是重新推断的分类，也不是 YOLO Pose 格式。
