# InfiniBand Multicast Performance Test

このディレクトリには、InfiniBandマルチキャストの性能テストプログラムが含まれています。複数ノードを跨いだマルチキャスト通信の性能を測定できます。

## 概要

このテストは以下の機能を提供します：

- InfiniBandデバイスの初期化
- UD（Unreliable Datagram）QPの作成と設定
- マルチキャストグループへの参加
- マルチキャストメッセージの送受信
- MPIを使用した複数ノードでのテスト実行
- **高性能なパフォーマンス測定機能**
- **詳細なログ出力とデータ整合性チェック**

## ファイル構成

- `ib_multicast_perf.c` - メインのパフォーマンステストプログラム
- `ib_ud_unicast_perf.c` - ユニキャスト性能テストプログラム
- `Makefile` - ビルド用Makefile
- `README.md` - このファイル

### ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照してください。 