install:
	pip install -U .
test: install
	py.test --pep8 .
