"""CLI entrypoint for full answer-supervision router training.

The reusable dataset, model, and expert-pair scoring implementation lives in
router_answer_supervision_core so cache construction does not import a trainer
script as a library.
"""

from router_answer_supervision_core import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
