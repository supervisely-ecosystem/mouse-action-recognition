mkdir models

wget https://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/u/K/Oj/ldq4nPukxMohitTytrINMQPbouqPPcgC6PpAYf1blLw4n8TptGc7c79C3wxPysH4fdtPpPAZUEDeFrQVSp89ILf9dV6Bh5LukLEsl23eAWNQJicbZ08HiSRSGsti.zip -O models/1089_RT-DETRv2.zip
wget https://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/teams_storage/P/p/YH/Qzh5UKv3EIGQZvaOX6G4IjcUZqmW8j7HxHoorXhXNYUU1tbwMD9mHeNQ4AiFMHirSyPumcRN8WMBlFXU8uY86rzDVS9yPteVRQN1s0ESNeO2sBUEW8H0UFqesFXD.zip -O models/MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26.zip

unzip models/1089_RT-DETRv2.zip -d models
unzip models/MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26.zip -d models

rm models/1089_RT-DETRv2.zip
rm models/MP_TRAIN_3_maximal_crop_2025-03-11_15-09-26.zip