export PYTHONPATH=../../../:PYTHONPATH

python clean_ipynb.py

# the exit code. set to 1 if one of the conversion fails
code=0
for python_args in `ls *.ipynb`
do
    echo ============================$python_args=============================
    jupyter nbconvert --execute --to notebook --inplace $python_args
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "The conversion failed with exit code $exit_code" >&2
        code=1
    fi
    echo ============================$python_args=============================
done

exit $code
