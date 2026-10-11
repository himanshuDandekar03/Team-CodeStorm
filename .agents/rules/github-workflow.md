# GitHub Workflow & Integration

The user's GitHub account is connected to Antigravity:
- **GitHub Username:** `himanshuDandekar03`
- **Name:** `Himanshu Dandekar`
- **Email:** `dandekarhimanshu2006@gmail.com`
- **CLI & Auth:** GitHub CLI (`gh`) and Git Credential Manager are authenticated and configured with repository permissions.

## When the user asks to "push the code to github" or similar:
1. **Ensure Git Repository is Initialized:**
   - If `.git` does not exist in the project root, run `git init -b main`.
   - Ensure sensitive files (`.env`, secrets, dependencies like `node_modules/`) are in `.gitignore`.
2. **Ensure Remote Origin Exists:**
   - Check with `git remote -v`.
   - If no remote is configured, create a new GitHub repository or connect an existing one using `gh`:
     ```powershell
     gh repo create <project-name> --public --source=. --remote=origin
     ```
     (or `--private` if the user specifies private).
3. **Stage, Commit, and Push:**
   - Stage changes: `git add -A`
   - Create a descriptive commit: `git commit -m "<concise commit message>"`
   - Push to GitHub: `git push -u origin <branch>`
