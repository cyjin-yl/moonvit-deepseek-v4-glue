# Qwen3B projector health probe v1

这 50 条样本沿用已冻结的 `screenspot_glm50_v1` 顺序。它们只用于训练过程中的
表征健康检查，不进入训练数据，也不能因为某个臂的结果不好而替换。每条记录同时
绑定 ScreenSpot 图片 SHA-256、缓存记录 SHA-256 和 MoonViT feature SHA-256。

前 100 个 optimizer step 使用全部 50 条缓存 feature 做 projector/receiver forward，
再对固定的前 8 条做 teacher-forced 的 vision、blind、shuffled 小探针。这里的
teacher-forced 结果是因果诊断，不能单独当作 ScreenSpot generation 成绩。

`PROBE_MANIFEST.json` 的 `manifest_sha256` 是去掉自身字段后的 canonical SHA-256；
`ARTIFACT_MANIFEST.json` 绑定文件大小和字节 SHA-256。
