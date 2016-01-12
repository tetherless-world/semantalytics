erase TensorExplorationPARAFAC.gv
erase TensorExplorationPARAFACjjs.gv
java -jar yesworkflow-0.2-SNAPSHOT-jar-with-dependencies.jar graph TensorExplorationPARAFACjjs.m
copy TensorExplorationPARAFACjjs.gv TensorExplorationPARAFAC.gv
dot -Tpdf TensorExplorationPARAFACjjs.gv -o TensorExplorationPARAFAC.pdf
