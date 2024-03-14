#  how to delete all commit history in github?

Deleting the .git folder may cause problems in your git repository. If you want to delete all your commit history but keep the code in its current state, it is very safe to do it as in the following:

+ 1) Checkout   
git checkout --orphan latest_branch  

+ 1) Add all the files  
git add -A  

+ 1) Commit the changes  
git commit -am "commit message"   

+ 1) Delete the branch  
git branch -D main  

+ 1) Rename the current branch to main
git branch -m main  

+ 1) Finally, force update your repository
git push -f origin main  

 