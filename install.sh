cd src
git clone https://github.com/dvmazur/mixtral-offloading.git --quiet
mv mixtral-offloading mixtral_offloading
cd mixtral_offloading && pip install -q -r requirements.txt
huggingface-cli download lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo --quiet --local-dir Mixtral-8x7B-Instruct-v0.1-offloading-demo