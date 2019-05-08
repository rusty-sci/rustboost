CA = cargo

runr:
	RUST_LOG=rustboost $(CA) run ./tests/data/generated/regression.libsvm -l regression

runc:
	RUST_LOG=rustboost $(CA) run ./tests/data/loaded/iris_classification.libsvm -l classification

runh:
	RUST_LOG=rustboost $(CA) run ./tests/data/generated/regression.libsvm -h

testlog:
	$(CA) test -- --nocapture

test:
	$(CA) test