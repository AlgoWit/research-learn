.PHONY: all clean test

clean:
	rm -rf coverage
	rm -rf dist
	rm -rf build
	rm -rf doc/_build
	rm -rf doc/auto_examples
	rm -rf doc/generated
	rm -rf doc/modules
	rm -rf examples/.ipynb_checkpoints

test-code:
	pytest rlearn

test-doc:
	pytest doc/*.rst

test-coverage:
	rm -rf coverage .coverage
	pytest --cov=rlearn rlearn

test: 
	make test-coverage test-doc

html:
	export SPHINXOPTS=-W; make -C doc html

code-format:
	black -S rlearn examples/*.py

code-analysis:
	flake8 --max-line-length=88 rlearn
	pylint -E rlearn/ -d E1103,E0611,E1101

upload-pypi:
	rm -rf ./build ./dist
	python setup.py sdist bdist_wheel
	twine upload dist/*

