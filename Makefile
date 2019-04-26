CA = cargo

run:
	RUST_LOG=rustboost $(CA) run ./tests/data/generated/regression.libsvm -l regression

runh:
	RUST_LOG=rustboost $(CA) run ./tests/data/generated/regression.libsvm -h

testlog:
	$(CA) test -- --nocapture

test:
	$(CA) test