# How to develop Renormalizer

The development protocol is very similar in all `Git` projects. 
For reference, you can check out the [development guide of NumPy](https://docs.scipy.org/doc/numpy/dev/).

## Development Process
You can hardly use `Git` effectively if you have no idea how it works. There are several ways to learn:
- Read the first 3 chapters of the [*Pro Git* book](https://git-scm.com/book/en/v2).
- Read all of [Git tutorial by Xuefeng Liao](https://www.liaoxuefeng.com/wiki/896043488029600) which is in Chinese.

---

Let's start with the first and most important rule:

**DO NOT PUSH TO THE MASTER BRANCH**.

Why? Keep on reading and you'll know. Actually, we've protected the master branch so that you can't push to it.
For a large project, this rule must be followed even for the owner of the project.

If nobody is allowed to push to the master branch, how is it possible to develop the branch?
The code in the master branch is updated by *merging* commits from other branches. 

Suppose now you have just cloned the whole package and are in the master branch, 
the first thing you need to do is to create a new branch for yourself and switch to that branch.
This can be done in a single command assuming you wish to name the new branch `develop`:
```
git checkout -b develop
```
Now you can make any modifications you like and commit them to your branch.
**After** you have **committed** all you want, you can merge all changes to the master following the procedures below:

1. Check if your branch is working by running `pytest` under the root directory.
1. Push your local branch to the `GitHub` server:
    ```
    git push origin develop
    ```
2. Create a *Pull Request (PR)* for your branch. This is done in the `GitHub` web pages. Near the button to choose the currently shown branches you can find another button called `New pull request`. Write helpful information when creating
the PR.
3. Check the status of the tests. These tests must all pass.
4. If anything went wrong with the tests or you want to further modify your codes, don't worry - it's perfectly common. 
You can continue your local development as usual and push them to your remote branch, after that, your PR will be automatically updated and all tests will be run again.
5. Wait for a reviewer to review and approve your code. Then the reviewer will merge your code for you if he's OK with your implementation.

Now you can do anything you want with your branch. Delete them, rename them, commit new codes, anything you like.
It's not related to the master branch anymore. 
You can continue your development on your branch and submit another PR after a while, 
or you can start a new branch.

You can compare this approach with the directly-push-to-master way. Which one is more safe and easier to manage if there are lots of contributors? IMO, the benefits of all the fuss are:
- The master branch is always stable. It is guaranteed that anyone at any time is able to get a working version of Renormalizer by cloning the master branch.
- The conflicts between different commits are mitigated. Suppose someone modified a lib function and commit to master before you, your commit might not work on the latest master branch although it might work on your local machine. This can be avoided with PR where the merged branch is also tested.
- You're allowed to modify the history of your commits. The history of the master branch shall not be modified but you can do anything you like in your own branch. 
- Changes in the code are dressed with contexts. In PRs there will be more explanations, discussions and cross-reference with other issues and PRs. This will help other people and your future self understand the code.

There is a [blog post](https://liwt31.github.io/2019/09/01/nopush/) on why you shouldn't push to master.

## Stylistic Guidelines
The style of the code (naming conventions, etc) follows [PEP 8](https://www.python.org/dev/peps/pep-0008/)
and you can find an explanation with better readability [here](https://pep8.org/). Some most important things:
- Use 4 spaces, not tab, for indentation. 
- Function names and variables should be lowercase, with words separated by underscores as necessary.
- Class names should normally use the CapWords convention.
- Imports should be grouped in the following order:
  1. Standard library imports.
  2. Related third party imports.
  3. Local application/library specific imports.

  You should put a blank line between each group of imports.

Automatic style checking is enabled in common IDEs such as PyCharm and there are [plugins for vim](https://github.com/nvie/vim-flake8). You can also format your code with [`Black`](https://github.com/psf/black).

## Common Mistakes during Development
A list of mistakes that the developers have made multiple times previously
- Forget to convert from float datatype to complex datatype. Usually NumPy will issue a warning.
