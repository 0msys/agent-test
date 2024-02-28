# このリポジトリについて

このリポジトリは[langchain](https://www.langchain.com/)のAgentを[Streamlit](https://streamlit.io/)と[Chainlit](https://docs.chainlit.io/get-started/overview)のそれぞれで実装したものです。

詳細は[こちらの記事](https://zenn.dev/0msys/articles/d5a97c8670d5fb)にあります。

## 動かし方

※ 動作させるには有効なOpenAIのAPIキーが必要です。

1. リポジトリをクローンし、`agent-test.env`というファイルを作成し、以下を記述して保存してください。
  
    ```env
    OPENAI_API_KEY="YOUR_API_KEY"
    ```

2. ファイルを用意したら、開発コンテナを起動してください。

3. 開発コンテナが起動したら、以下のコマンドでそれぞれ実行してください。

   - Streamlitを実行する場合
  
      ```bash
      streamlit run src/streamlit.py
      ```

   - Chainlitを実行する場合
  
      ```bash
      chainlit run src/chainlit.py
      ```

4. ブラウザでにアクセスしてください。
   - Streamlit: `http://localhost:8501`
   - Chainlit: `http://localhost:8000`

