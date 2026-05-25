# Google drive links to download the training data
gdown https://drive.google.com/uc?id=1BuKEc9qoBVgF8ApTTgAKRFVvDagWHPt7 # hamer_training_data_part1.tar.gz
gdown https://drive.google.com/uc?id=1lNqBsifaxMP3NHIV_KKJVCT1zDvUn2T_ # hamer_training_data_part2.tar.gz
gdown https://drive.google.com/uc?id=16xfV_ALY_M3VZeXKpjlB3MhnCNiS5XSq # hamer_training_data_part3.tar.gz
gdown https://drive.google.com/uc?id=1SqzFHH2-UI6PlGTMd0Ds2FJKALBBOaEO # hamer_training_data_part4a.tar.gz
gdown https://drive.google.com/uc?id=1xQxEjpaa3WqJt60UMtX_wpnOZ3bpj-u9 # hamer_training_data_part4b.tar.gz
gdown https://drive.google.com/uc?id=1ozejeAue1M-p4boIfpnpX0E8RFIeHUZH # hamer_training_data_part4c.tar.gz

# Alternatively, consider using the dropbox links:
#wget -O hamer_training_data_part1.tar.gz https://www.dropbox.com/scl/fi/bqq0jheev3626q1wiijs1/hamer_training_data_part1.tar.gz?rlkey=8fv4ktvk7r3txofd90q0trgxr
#wget -O hamer_training_data_part2.tar.gz https://www.dropbox.com/scl/fi/l9l5udalchu0mh4qxnw2t/hamer_training_data_part2.tar.gz?rlkey=i0n2lzix4q6jxmhm4sr5rtmkt
#wget -O hamer_training_data_part3.tar.gz https://www.dropbox.com/scl/fi/6lamcbwt79ri0oj4knwm3/hamer_training_data_part3.tar.gz?rlkey=j5y7ea7xrlu440ud12otaj2ne
#wget -O hamer_training_data_part4a.tar.gz https://www.dropbox.com/scl/fi/vp6cw7he8t0eigjf6001l/hamer_training_data_part4a.tar.gz?rlkey=wylmufft4a5nq3yxep2olifrk
#wget -O hamer_training_data_part4b.tar.gz https://www.dropbox.com/scl/fi/vyjasngr67ru14fb8s108/hamer_training_data_part4b.tar.gz?rlkey=qgotg1v9lkgo5eu78gh8b007t
#wget -O hamer_training_data_part4c.tar.gz https://www.dropbox.com/scl/fi/nfvz5zpcmhz8hkwzc6ji4/hamer_training_data_part4c.tar.gz?rlkey=ygh0wvse04twhh1ri3xiw2sag

for f in hamer_training_data_part*.tar.gz; do
    tar --warning=no-unknown-keyword --exclude=".*" -xvf $f
done
