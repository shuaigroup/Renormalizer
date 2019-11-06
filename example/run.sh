export PYTHONPATH=../:PYTHONPATH
code=0
for python_args in fmo.py sbm.py "transport.py std.yaml" "transport_autocorr.py std.yaml"; do
    echo ============================$python_args=============================
    timeout 20s python $python_args
    exit_code=$?
    echo ============================$python_args=============================
    # the time out exit code
    if [ $exit_code -ne 124 ]; then
        echo "The script failed with exit code $exit_code" >&2
        code=1
    fi
done

exit $code