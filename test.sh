#!/usr/bin/env bash
pycodestyle --max-line-length=120 torch_same_pad tests && \
    nosetests --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=torch_same_pad --with-doctest