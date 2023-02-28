.PHONY: clean

cases = c1 c7_fine c7_coarse pipeFlowCoarse  pipeFlowFine

clean:
	for c in $(cases); do python -m pintFoam.clean $$c; done

