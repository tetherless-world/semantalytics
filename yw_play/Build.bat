erase CPPData.gv
erase CPPDatajjs.gv
java -jar yesworkflow-0.2-SNAPSHOT-jar-with-dependencies.jar graph CPPDatajjs.m
copy CPPDatajjs.gv CPPData.gv
dot -Tpdf CPPData.gv -o CPPData.pdf
