class Hparams:
    batch_size = 256
    enc_maxlen = 30*2
    dec_maxlen = 40*2
    num_epochs = 50*2
    hidden_units = 64*8
    emb_units = 64*4
    graphemes = ["<pad>", "<unk>", "</s>"] + list('พจใงต้ืฮแาฐฒฤๅูศฅถฺฎหคสุขเึดฟำฝยลอ็ม ณิฑชฉซทรฏฬํัฃวก่ป์ผฆบี๊ธญฌษะไ๋นโภ?')
    phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + list('-พจใงต้ืฮแาฐฒฤูศฅถฺฎหคสุขเึดฟำฝยลอ็ม ณิฑชฉซทรํฬฏ–ัฃวก่ปผ์ฆบี๊ธฌญะไษ๋นโภ?')
    lr = 0.001

hp = Hparams()