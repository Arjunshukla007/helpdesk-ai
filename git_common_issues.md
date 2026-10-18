# Git Common Issues

## Undo the last commit
Use `git reset --soft HEAD~1` to undo the last commit but keep your changes staged. Use `git reset --hard HEAD~1` only if you want to discard the changes completely.

If the commit was already pushed, use `git revert <commit>` instead so shared history is not rewritten.

## Merge conflicts
Open the conflicted files and look for the markers `<<<<<<<`, `=======` and `>>>>>>>`. Edit the file to keep the correct code, then run `git add <file>` and `git commit`.

To cancel a merge that went wrong, run `git merge --abort`.

## Committed to the wrong branch
Create a new branch from the current state with `git branch fix-branch`, then move the original branch back using `git reset --hard origin/main`.
