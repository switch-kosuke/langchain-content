# Create Agent App with GenerativeAI

## About Agent App
本セッションでは、LinkedInの情報を要約して表示する。  

### スクレイピングした情報からLLMに回答してもらう（Linkedin.py）
- Linkedinをスクレイピングする  
    Linkedinの標準APIは、扱いが難しい。  
    そこで、今回は[Proxycurl](https://nubela.co/proxycurl/linkedin)を用いて、スクレイピングする。  

- スクレイピングしたjsonをきれいにする
    jsonをきれいにするために、[Jsonlint](https://jsonlint.com/)を用いた。（これは何でもいい）

    ちなみに、スクレイピングしたjsonは、[GitHub gist](https://gist.github.com/)で公開することもできる。

- これらのスクレイピングした情報を、Les1で作成したLangchainにinformationとして付加することで完成  

