# Install dependencies
dep:
	pip install -r requirements.txt

# Install developer dependencies
dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Format code with black and isort
format:
	black .
	seed-isort-config
	isort -y

# Test code with black, flake8, isort, mypy, and pytest.
test:
	black --check .
	flake8
	isort **/*.py -c
	mypy .
	pytest
