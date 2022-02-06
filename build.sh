#!/bin/bash
for cmd in "rm -rf _build" "jupyter-book build ." "perl -pi -e 's{https://mybinder.org/v2/gh/wangxuuu/Smoothed-InfoNCE/main\?urlpath=tree/(.*\.ipynb)}{https://mybinder.org/v2/gh/ccha23/binder/master?urlpath=git-pull?repo%3Dhttps%3A%2F%2Fgithub.com%2Fwangxuuu%2FSmoothed-InfoNCE%26urlpath%3Dtree%2FSmoothed-InfoNCE%2F\$1%26branch%3Dmain}g' _build/html/*.html" "ghp-import -n -p -f _build/html"
do
    read -r -p "${cmd}?[Y/n] " input

    case $input in
        [yY][eE][sS]|[yY]|'')
    echo "Executing..."
    eval $cmd
    ;;
        [nN][oO]|[nN])
    echo "Skipped..."
        ;;
        *)
    echo "Invalid input..."
    exit 1
    ;;
    esac
done
