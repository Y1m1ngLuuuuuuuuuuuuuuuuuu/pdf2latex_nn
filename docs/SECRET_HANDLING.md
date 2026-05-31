# Secret Handling

Local secrets must never be committed, copied into canonical evidence, or placed
inside migration/share packages.

Known local secret-risk file after the post-submission audit:

```text
.env.local
```

Use `.env.example` and `config/paths.local.template.yaml` as templates only.
Real credentials, AutoDL passwords, Kaggle tokens, OpenAI keys, access tokens,
and private keys belong in ignored local files outside any shared package.

