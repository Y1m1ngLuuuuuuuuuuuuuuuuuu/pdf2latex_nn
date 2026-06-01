# Secret Handling

Local secrets must never be committed, copied into canonical evidence, placed
inside migration/share packages, or uploaded with runtime tarballs.

Known private local area after reorganization:

```text
/Users/lu/Code/Project/pdf2latex_nn/private_config_do_not_upload/
```

Known local secret-risk file:

```text
.env.local
```

Use these templates only:

```text
.env.example
config/paths.local.template.yaml
```

Never commit or package:

- real credentials
- AutoDL passwords
- Kaggle tokens
- OpenAI/API keys
- access tokens
- private SSH keys
- rclone or netdisk private configs

Runtime backups and GitHub source packages must exclude private config files.
