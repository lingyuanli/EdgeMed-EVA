# M6 option-text likelihood pilot64：FAILED

日期：2026-09-04  
层级：PMC external development；不是 Med-CMR official score

## 合同与运行

- Qwen3.5-4B base，NF4 / FP16 / eager，图像上限 786,432 pixels。
- user prompt 只含图像与问题；每个完整选项文本分别作为 assistant completion。
- score：包含消息结束 token 的平均 completion log-probability。
- 4 条 smoke：4/4，20/20 finite option scores，0 invalid。
- 64 条 pilot：64/64，`67.27 s`，峰值 `4,744.79 MiB`。
- contract SHA-256：`392b610a33913456e3d4ec74856c031bbc1a77d516692e096d6d9b348b49c492`。
- predictions SHA-256：`6abe680b68f4110069c1aff6a2b795aae66d67448efd69db2c3aa6f6ab0a14fb`。
- subset references SHA-256：`8d438942ce29c3e61ea30b5f4ee0bd378d82a018cd8295a7f826354bf8465c40`，mode 0600。

## 配对结果

| Method | Correct | Accuracy | Invalid |
|---|---:|---:|---:|
| direct B0 | 37/64 | 57.81% | 1/64 |
| M6 likelihood | 27/64 | 42.19% | 0/64 |

M6-direct 为 `-15.63` points，paired bootstrap 95% CI `[-28.13, -3.13]`，McNemar exact `p=0.0414`；contingency 为 B0-only 15、M6-only 5、both-correct 22、both-wrong 22。

## 判定

预注册的 `+3.0` point 扩大门显著失败。0 invalid 只证明构造性格式有效，不能支持答案性能。停止完整 512；不根据本次逐题正确性修改长度归一化、EOM、prompt 或候选集合。若未来采用判别式 MCQ，需把该分数作为受监督训练目标并重新立项。
