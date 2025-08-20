import torch
sd = torch.load("fingerprints/lp/lp_citeseer.pt", map_location="cpu")
st = sd.get("state_dict", sd)
enc_only = {k.replace("enc.", "", 1): v for k, v in st.items() if k.startswith("enc.")}
torch.save({"state_dict": enc_only, "arch":"gcn", "in_dim": sd["in_dim"], "hid": 64, "out": 64, "layers": 3},
           "fingerprints/lp/enc_citeseer.pt")
print("wrote fingerprints/lp/enc_citeseer.pt")
