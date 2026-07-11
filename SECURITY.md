# Security Policy

This is a portfolio project and does not include production authentication, authorization, or user management.

## Credentials

Do not commit:

- Kaggle credentials.
- `.env` files.
- API keys.
- Tokens.
- Passwords.
- Local access files.

The repository `.gitignore` excludes common credential and environment files.

## Reporting Issues

If you find a secret committed by mistake or a security concern in the project, open a private communication channel with the repository owner before creating a public issue.

## Scope

The Streamlit dashboard is intended for local portfolio demonstration. It is not hardened for public multi-user production deployment.
