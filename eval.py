"""Command-line entry point for EvalMono3D (thin wrapper over ``evalmono3d.cli``).

Example::

    python eval.py \\
        --name cdrone_test_example \\
        --gt_ann  example/cdrone_test_example.json \\
        --pred_ann example/cdrone_test_example_pred.pth \\
        --log_dir /tmp/eval_cdrone_test_example
"""

from evalmono3d.cli import main

if __name__ == "__main__":
    main()
