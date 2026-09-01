# PMC-VQA v2 Cycle-1 Admission Receipt

Run date: 2026-09-01  
Execution host: `ubuntu@117.50.188.27`  
Source root: `/home/ubuntu/data/external/pmc-vqa-b56ae594`

## Source binding

- dataset revision: `b56ae594f794867893143b337b4118a835794647`
- `train_2.csv`: 56,703,454 bytes, SHA-256
  `15ead4a27b5365d1b5dc5faf1fd2246a6a5aa1ca5c2358a19532ff78c2870084`
- `test_2.csv`: 12,446,345 bytes, SHA-256
  `d57d567f997955e7001ec4325323a2bca66d570070466291d5ec12882ff5ba09`
- `oa_comm_use_file_list.csv`: 496,021,536 bytes, SHA-256
  `7a56eef6527332d2eb47fffa1356ffe19548b8984823a533d174fd70bb38b86f`
- `images_2.zip`: 2,206,255,503 bytes, SHA-256
  `727643d0ae9182cb5572b43a74ed4100eb0920b1d83e55ed4054a393d672cb4a`
- safe extraction: 164,360 files / 2,243,427,109 uncompressed bytes
- extraction report SHA-256:
  `7494843f944c94f8c779f6b7aed7a25927a6c1aa72eecebc6804b40f1455bef6`

The eligible source rows are overwhelmingly CC BY with a small CC0 subset.
Rows missing a PMCID/license join are excluded. CC BY-ND is not admitted.

## Training seed

- deterministic source selection: 2,000 rows, 2,000 unique images, 1,945 articles
- source build report SHA-256:
  `746b2310e891dc710679c81025e35a605f2cb2c7debcb897922d553f2c1ab152`
- pre-quarantine manifest SHA-256:
  `2e33852b0bdaf467f303aacd305c22314f9151bf2acb3c0d91829dab43f6fc9c`
- first gate: 0 confirmed overlaps; 185 dHash candidates across 32 external images
- policy: no manual override for the training pool; all 32 records were marked
  `quarantined/suspected`
- admitted manifest: 1,968 accepted + 32 retained quarantined records
- admitted manifest SHA-256:
  `c19c67e0746da3756518c763168d51a54b749da012cb97305e5d0a2c58053a8d`
- quarantine report SHA-256:
  `b7cf4e244bc251137214a43393a5f51418bfdd15c796468a3d80b5b8a546f2f2`
- final gate: `passed`; 1,968 checked, 32 skipped, 0 file problems,
  0 confirmed overlaps, 0 remaining candidates
- final gate report SHA-256:
  `addae7b0b1ccc107aeae497476aace87661a49df6f805f13e154f23c7e2b364d`

## MCQ development cohort

- deterministic source selection: 512 rows, 512 unique images, 493 articles
- train/dev article overlap: 0
- train/dev image SHA-256 overlap: 0
- normalized exact question overlap after exclusion: 0
- source build report SHA-256:
  `abe9337d08f2bd833338e30570381efa67231a66cb637c5119629740d873028f`
- manifest SHA-256:
  `c5fb9dabcff592be7e8c558f2eab515dd9f459685d04bc37ad2cdf0d51e48b05`
- gate: `passed`; 512 checked, 0 file problems, 0 confirmed overlaps
- dHash candidates: 15 unique pairs across 5 external images; all were visually
  audited as distinct images and their maximum pixel correlation was 0.8108015
- gate report SHA-256:
  `24b45520a82ef946fe5150b7b2dc27608e62b15491b847a1d13aecd0662e8839`

Both gates used only the answer-free 16,655-row Med-CMR MCQ manifest at SHA-256
`9ec6f833f1f53509d25873b2beb77960f18d55b4b514a0f4796efd147d0219d7`.
No Med-CMR references, predictions, or scores entered source selection.

## Allowed use

- 1,968 accepted train rows: T1a answer-only QLoRA; captions are retained for
  audit but are not eligible evidence targets.
- 512 dev rows: primary direct-versus-B1 MCQ comparison and checkpoint selection.
- Med-CMR: remains final frozen test only.

