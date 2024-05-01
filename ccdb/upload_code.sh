#zip -r ../* oneiros.zip
zip -r oneiros.zip ../../oneiros -x "../../oneiros/venv/*" \
  "../../oneiros/external/*" "../../oneiros/plots/*" \
  "../../oneiros/ccdb/oneiros.sif" "../../oneiros/outputs/*" \
  "../../oneiros/multirun/*" "../../oneiros/wandb/*" "../../oneiros/\.git/*" \
  "../../oneiros/\.idea/*" "../../oneiros/ccdb/*"
scp -r ./oneiros.zip cgauthie@narval.computecanada.ca:/home/cgauthie/projects/def-lpaull/cgauthie/oneiros/code.zip