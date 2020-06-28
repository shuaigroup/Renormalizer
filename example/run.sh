export PYTHONPATH=../:PYTHONPATH
code=0
for python_args in fmo.py sbm.py h2o_qc.py "dynamics.py std.yaml" "transport_kubo.py std.yaml"; do
    echo ============================$python_args=============================
    timeout 20s python $python_args
    exit_code=$?
    echo ============================$python_args=============================
    # if not the time out exit code or normal exit code
    if [ $exit_code -ne 124 ] && [ $exit_code -ne 0 ]; then
        echo "The script failed with exit code $exit_code" >&2
        code=1
    fi
done

exit $code
