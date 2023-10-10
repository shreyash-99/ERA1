# ERAv1_Session16

This repository contains the PyTorch implementation of the Original Transformer Paper: [Transformer](https://arxiv.org/pdf/1706.03762.pdf), just with a few strategies that would speed up the training process.

Refer to [S16.ipynb](Session16_AssignmentSolution.ipynb) jupyter notebook for the training steps.

#### Strategies Used:
1. AMP [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
2. OneCyclePolicy to find the optimum learning rate and no. of epochs needed to reach it.
3. Dynamic padding for effectively reduce the training time.
4. Removed all english sentences with token > 120

#### Output Achieved
Reached at a loss 1.743 in just 15 epochs using above techniques.<br>
Reduced training time by half(could have reduced by much more by increased the batch size from 6 to 8)

#### One Cycle Policy
OCP (One Cycle Policy) was used to get the optimum learning_rate.
```
scheduler = OneCycleLR(
    optimizer,
    max_lr=1E-04,
    steps_per_epoch=len(train_data_loader),
    epochs=40,
    pct_start=0.125,
    div_factor=10,
    three_phase=True,
    final_div_factor=10,
    anneal_strategy='linear'
)
```

#### Automatic Mixed Precision
Used at the respective steps mentioned below
```
scaler = torch.cuda.amp.GradScaler()

    scaler.scale(loss).backward()

    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skip_lr_sched = (scale > scaler.get_scale())

```

#### Dynamic Padding
Used collate function when loading texts and forming batches, kept sequence_len as the max length of a sentence passed in a batch.<br>
Our transformer architecture doesnt depend on seq_len therefor we can use this to save on computing power. 
```
def collate_fn(batch):
    encoder_input_max = max(b['encoder_str_length'] for b in batch)
    decoder_input_max = max(b['decoder_str_length'] for b in batch)
    encoder_input_max += 2
    decoder_input_max += 2

    # input_size_max = max(encoder_input_max, decoder_input_max)

    pad_token_encoder = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype = torch.int64)
    pad_token_decoder = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype = torch.int64)

    encoder_inputs = []
    decoder_inputs = []
    encoder_masks = []
    decoder_masks = []
    labels = []
    src_texts = []
    tgt_texts = []

    for b in batch:
        enc_num_padding_token = encoder_input_max - len(b['encoder_input'])
        dec_num_padding_token = decoder_input_max - len(b['decoder_input'])
        label_num_padding_token = decoder_input_max - len(b['label'])

        encoder_input = torch.cat(
            [
                b['encoder_input'],
                torch.tensor([pad_token_encoder] * enc_num_padding_token , dtype = torch.int64)
            ],
            dim = 0,
        )
        decoder_input = torch.cat(
            [
                b['decoder_input'],
                torch.tensor([pad_token_decoder] * dec_num_padding_token, dtype = torch.int64)
            ],
            dim = 0,
        )
        label = torch.cat(
            [
                b['label'],
                torch.tensor([pad_token_decoder] * label_num_padding_token, dtype = torch.int64)
            ],
            dim = 0,
        )
        encoder_mask = (encoder_input != pad_token_encoder).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input != pad_token_decoder).unsqueeze(0).int() & causal_mask(decoder_input_max)
        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)
        encoder_masks.append(encoder_mask)
        decoder_masks.append(decoder_mask)
        labels.append(label)
        src_texts.append(b["src_text"])
        tgt_texts.append(b['tgt_text'])

    return {
        "encoder_input": torch.vstack(encoder_inputs),
        "decoder_input": torch.vstack(decoder_inputs),
        "encoder_mask": torch.vstack(encoder_masks),
        "decoder_mask": torch.vstack(decoder_masks),
        "label" : torch.vstack(labels),
        "src_text" : src_texts,
        "tgt_text": tgt_texts
    }
```
#### Removing long sentences
```
    ## keep 90% for traning and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    sorted_train_ds = sorted(train_ds_raw, key = lambda x:len(x["translation"][config['lang_src']]))
    # sorted_train_ds = train_ds_raw ## not sorted, taken as it is
    filtered_sorted_train_ds = [k for k in sorted_train_ds if (len(k['translation'][config['lang_src']]) < 150 and  len(k['translation'][config['lang_src']]) > 3)]
    filtered_sorted_train_ds = [k for k in filtered_sorted_train_ds if (len(k['translation'][config['lang_tgt']]) < 150 and len(k['translation'][config['lang_tgt']]) > 3)]
    filtered_sorted_train_ds = [k for k in filtered_sorted_train_ds if len(k['translation'][config['lang_src']]) + 10 > len(k['translation'][config['lang_tgt']]) ]
```


#### Training Logs
```
Using device:  cuda
dataset_size 127085
Max length of source sentence: 471
Max length of target sentence: 482
Max length of filtered source sentence: 45
Max length of filterd target sentence: 48
length of train dataset 60888
length of validation dataset 12709
Total Parameters: 57124690
Processing Epoch 00: 100%|██████████| 7611/7611 [17:14<00:00,  7.36it/s, loss_acc=4.823, loss=4.476, lr=0.00039989488437281004]
--------------------------------------------------------------------------------
    SOURCE: Unable to comprehend the captain's resistance, he hastened to say to him,−−
    TARGET: Ne pouvant s’expliquer la résistance du capitaine, il se hâta de lui dire :
 PREDICTED: Le capitaine voulut lui comprendre le capitaine , il se hâta de lui dire :
--------------------------------------------------------------------------------
    SOURCE: The elder one, whom you have seen (and whom I cannot hate, whilst I abhor all his kindred, because he has some grains of affection in his feeble mind, shown in the continued interest he takes in his wretched sister, and also in a dog-like attachment he once bore me), will probably be in the same state one day.
    TARGET: L'aîné, que vous avez vu (et que je ne puis pas haïr, bien que je déteste toute sa famille, parce que cet esprit faible a montré, par son continuel intérêt pour sa malheureuse soeur, qu'il y avait en lui quelque peu d'affection, et parce qu'autrefois il a eu pour moi un attachement de chien), aura probablement, un jour à venir, le même sort que les autres;
 PREDICTED: Le bon , que vous m ’ avez vu , et que je ne puis pas , je ne le , car il me semble que de faire en faire l ’ esprit .
--------------------------------------------------------------------------------
Processing Epoch 01: 100%|██████████| 7611/7611 [19:14<00:00,  6.59it/s, loss_acc=3.801, loss=3.257, lr=0.0006997897687456201]
--------------------------------------------------------------------------------
    SOURCE: « Si cependant vous avez la bonté de me mettre au courant de vos investigations, continua-t-il, je serai heureux de vous preter mon concours dans la limite de mes moyens.
    TARGET: "If you will let me know how your investigations go," he continued, "I shall be happy to give you any help I can.
 PREDICTED: " If you have done my of my , I ' s , I ," he continued , " I am , " I am in my of my ."
--------------------------------------------------------------------------------
    SOURCE: "No doubt it is a geyser, like those in Iceland."
    TARGET: --Eh! sans doute, geyser, riposte mon oncle, un geyser pareil à ceux de l'Islande!»
 PREDICTED: -- Non , c ' est un homme , comme ceux dans l ' Islande .
--------------------------------------------------------------------------------
Processing Epoch 02: 100%|██████████| 7611/7611 [19:10<00:00,  6.62it/s, loss_acc=3.460, loss=3.657, lr=0.000999645234758234]
--------------------------------------------------------------------------------
    SOURCE: The fair Amanda reflected for a while.
    TARGET: La belle Amanda réfléchit un peu.
 PREDICTED: L ’ Amanda songea pour un temps .
--------------------------------------------------------------------------------
    SOURCE: But his thoughts had never turned in that direction, and, moreover, he had not the least inclination for riotous living.
    TARGET: Il n'y avait pas pensé, parce que sa chair était morte, et qu'il ne se sentait plus le moindre appétit de débauche.
 PREDICTED: Mais sa pensée n ' avait pas dans cette direction , et , et , il n ' avait pas le moins de .
--------------------------------------------------------------------------------
Processing Epoch 03: 100%|██████████| 7611/7611 [19:06<00:00,  6.64it/s, loss_acc=3.113, loss=3.385, lr=0.0007004598808689559]
--------------------------------------------------------------------------------
    SOURCE: A noise aroused him; someone was knocking at the door, trying to open it.
    TARGET: Un bruit le réveilla, on frappait a la porte, on essayait d'ouvrir.
 PREDICTED: Un bruit le fit ; quelqu ’ un s ’ en allait , en s ’ en .
--------------------------------------------------------------------------------
    SOURCE: When they reached the shop, everyone was ready: Grivet and Olivier, the witnesses of Therese, were there, along with Suzanne, who looked at the bride as little girls look at dolls they have just dressed up.
    TARGET: Lorsqu'ils arrivèrent à la boutique, tout le monde était prêt: il y avait là Grivet et Olivier, témoins de Thérèse, et Suzanne qui regardait la mariée comme les petites filles regardent les poupées qu'elles viennent d'habiller.
 PREDICTED: Quand ils furent prêts à la boutique ; chacun était prêt à Olivier et Olivier , la témoins , il y avait des filles , qui regardait de l ' avoir des .
--------------------------------------------------------------------------------
Processing Epoch 04: 100%|██████████| 7611/7611 [19:48<00:00,  6.40it/s, loss_acc=2.633, loss=3.261, lr=0.00040060441485634203]
--------------------------------------------------------------------------------
    SOURCE: « Voila mon dernier anneau, cria-t-il, tout est complet maintenant. »
    TARGET: "The last link," he cried, exultantly. "My case is complete."
 PREDICTED: " That last the last ring ," he cried , " is his astonishment ."
--------------------------------------------------------------------------------
    SOURCE: CHAPTER 16
    TARGET: CHAPITRE XVI.
 PREDICTED: CHAPITRE XVI
--------------------------------------------------------------------------------
Processing Epoch 05: 100%|██████████| 7611/7611 [19:55<00:00,  6.37it/s, loss_acc=2.229, loss=2.265, lr=0.0001007489488437281]
--------------------------------------------------------------------------------
    SOURCE: If your feelings are still what they were last April, tell me so at once. _My_ affections and wishes are unchanged, but one word from you will silence me on this subject for ever."
    TARGET: Les miens n’ont pas varié, non plus que le reve que j’avais formé alors. Mais un mot de vous suffira pour m’imposer silence a jamais.
 PREDICTED: Si vos sentiments sont encore si elles me le dire , si je suis incapable de me rendre la route , mais un mot de silence pour vous , je vous réponds sur ce sujet .
--------------------------------------------------------------------------------
    SOURCE: "So did I, madam, and I am excessively disappointed.
    TARGET: -- Moi aussi, madame, et vous me voyez très désappointé.
 PREDICTED: -- Je m ' a dit , madame , et je suis un peu longue .
--------------------------------------------------------------------------------
Processing Epoch 06: 100%|██████████| 7611/7611 [19:41<00:00,  6.44it/s, loss_acc=1.992, loss=2.046, lr=9.626086004434347e-05]
--------------------------------------------------------------------------------
    SOURCE: Observing Conseil, I discovered that, just barely, the gallant lad had fallen under the general influence.
    TARGET: En observant Conseil, je constatai que ce brave garçon subissait tant soit peu l'influence générale.
 PREDICTED: Je , Conseil , à peine , à peine , que le brave garçon était tombé sous l ' influence général .
--------------------------------------------------------------------------------
    SOURCE: "I tell you that she can't see you."
    TARGET: -- Je n'y peux rien, s'écria la femme d'un ton rude, je vous répète qu'elle ne peut vous voir.
 PREDICTED: -- Je vous dis qu ' elle ne peut pas vous voir .
--------------------------------------------------------------------------------
Processing Epoch 07: 100%|██████████| 7611/7611 [19:46<00:00,  6.42it/s, loss_acc=1.936, loss=1.878, lr=9.25123586894041e-05]
--------------------------------------------------------------------------------
    SOURCE: Among some unimportant papers he found the following letter, that which he had sought at the risk of his life:
    TARGET: Au milieu de quelques papiers sans importance, il trouva la lettre suivante: c'était celle qu'il était allé chercher au risque de sa vie:
 PREDICTED: Quelques papiers , après avoir trouvé la lettre , ce qui était essayé de courir au risque de sa vie :
--------------------------------------------------------------------------------
    SOURCE: Thereupon her son had a nervous attack, and threatened to fall ill, if she did not give way to his whim.
    TARGET: Son fils eut une crise de nerfs, il la menaça de tomber malade, si elle ne cédait pas à son caprice.
 PREDICTED: Là - dessus son fils avait une crise nerveuse , et il menaçait de ne pas se laisser agir franchement , de sa fantaisie .
--------------------------------------------------------------------------------
Processing Epoch 08: 100%|██████████| 7611/7611 [19:46<00:00,  6.42it/s, loss_acc=1.892, loss=1.719, lr=8.876336462923932e-05]
--------------------------------------------------------------------------------
    SOURCE: Mrs. Fairfax turned out to be what she appeared, a placid-tempered, kind-natured woman, of competent education and average intelligence.
    TARGET: Mme Fairfax était en effet ce qu'elle m'avait paru tout d'abord, une femme douce, complaisante, suffisamment instruite, et d'une intelligence ordinaire.
 PREDICTED: Mme Fairfax se tourna vers cette éducation de l ' éducation tranquille et capables de l ' éducation de capables , de l ' intelligence et de toute intelligence .
--------------------------------------------------------------------------------
    SOURCE: The host drew back and burst into tears.
    TARGET: L'hôte recula d'un pas et se mit à fondre en larmes.
 PREDICTED: L ' hôte se recula , et éclata en sanglots .
--------------------------------------------------------------------------------
Processing Epoch 09: 100%|██████████| 7611/7611 [18:47<00:00,  6.75it/s, loss_acc=1.856, loss=2.037, lr=8.501486327429995e-05]
--------------------------------------------------------------------------------
    SOURCE: I thought I recognised you at street-corners, and I ran after all the carriages through whose windows I saw a shawl fluttering, a veil like yours."
    TARGET: J’ai cru vous reconnaître au coin des rues; et je courais après tous les fiacres où flottait à la portière un châle, un voile pareil au vôtre...
 PREDICTED: Je croyais que vous dans les coin de la rue ; et je courus de toute la voiture des voitures au un voile , bien tourner la vôtre .
--------------------------------------------------------------------------------
    SOURCE: ÉTIENNE had at last descended from the platform and entered the Voreux; he spoke to men whom he met, asking if there was work to be had, but all shook their heads, telling him to wait for the captain.
    TARGET: Étienne, descendu enfin du terri, venait d'entrer au Voreux; et les hommes auxquels il s'adressait, demandant s'il y avait du travail, hochaient la tete, lui disaient tous d'attendre le maître-porion.
 PREDICTED: Des descendirent sur la plate - forme et , il rentra au Voreux ; il a des hommes qu ' il rencontrait , s ' il n ' y avait que la tete .
--------------------------------------------------------------------------------
Processing Epoch 10: 100%|██████████| 7611/7611 [18:49<00:00,  6.74it/s, loss_acc=1.826, loss=1.917, lr=8.126636191936059e-05]
--------------------------------------------------------------------------------
    SOURCE: The porter said, 'Yes, madam'; and the constable began not to like it, and would have persuaded the mercer to dismiss him, and let me go, since, as he said, he owned I was not the person.
    TARGET: Le commissionnaire dit: «Oui, madame»; et la chose commença de déplaire au commissaire qui s'efforça de persuader au mercier de me congédier et de me laisser aller, puisque, ainsi qu'il disait, il convenait que je n'étais point la personne.
 PREDICTED: Le portier , madame . Et le commissaire se mit à commissaire en commissaire , et ne l ' eût pas dit le mercier , car je fus reçue , et me , car il ne le but , car il ne pouvait .
--------------------------------------------------------------------------------
    SOURCE: "Are you a Christian?"
    TARGET: -- Êtes-vous chrétien?
 PREDICTED: -- Vous êtes un chrétien ?
--------------------------------------------------------------------------------
Processing Epoch 11: 100%|██████████| 7611/7611 [18:45<00:00,  6.76it/s, loss_acc=1.802, loss=1.787, lr=7.751835326964662e-05]
--------------------------------------------------------------------------------
    SOURCE: "Listen!" he said to her; and she shuddered at the sound of that fatal voice which she had not heard for a long time.
    TARGET: « Écoute », lui dit-il, et elle frémit au son de cette voix funeste qu’elle n’avait pas entendue depuis longtemps. Il continua.
 PREDICTED: « Écoutez , lui dit - il , et elle frissonnait au son de cette voix fatal qui ne l ' avait pas entendu pendant longtemps .
--------------------------------------------------------------------------------
    SOURCE: At eight o'clock Justin came to fetch him to shut up the shop.
    TARGET: À huit heures, Justin venait le chercher pour fermer la pharmacie.
 PREDICTED: À huit heures , Justin venait l ’ aller chercher pour qu ’ il fermait la boutique .
--------------------------------------------------------------------------------
Processing Epoch 12: 100%|██████████| 7611/7611 [18:50<00:00,  6.74it/s, loss_acc=1.779, loss=1.795, lr=7.376985191470725e-05]
--------------------------------------------------------------------------------
    SOURCE: "That was what I wished you to think."
    TARGET: – C’est ce que je désirais vous faire croire.
 PREDICTED: -- C ' est ce que je voulais vous croire .
--------------------------------------------------------------------------------
    SOURCE: Eight o'clock struck.
    TARGET: Huit heures sonnaient.
 PREDICTED: Huit heures sonnerent .
--------------------------------------------------------------------------------
Processing Epoch 13: 100%|██████████| 7611/7611 [18:41<00:00,  6.79it/s, loss_acc=1.760, loss=1.979, lr=7.002184326499329e-05]
--------------------------------------------------------------------------------
    SOURCE: "Yes, but water decomposed into its primitive elements," replied Cyrus Harding, "and decomposed doubtless, by electricity, which will then have become a powerful and manageable force, for all great discoveries, by some inexplicable laws, appear to agree and become complete at the same time.
    TARGET: -- Oui, mais l'eau décomposée en ses éléments constitutifs, répondit Cyrus Smith, et décomposée, sans doute, par l'électricité, qui sera devenue alors une force puissante et maniable, car toutes les grandes découvertes, par une loi inexplicable, semblent concorder et se compléter au même moment.
 PREDICTED: -- Oui , mais dans ses éléments Bunzen , répondit Cyrus Smith , et il aura été par l ' électricité qui se tout à fait pour un grand fracas .
--------------------------------------------------------------------------------
    SOURCE: En effet, j’ai vu deux de ses espions particuliers, de moi bien connus, se promener dans ma rue jusque sur le minuit.
    TARGET: As a matter of fact, I saw two of his private spies, well known to me, patrolling my street until nearly midnight.
 PREDICTED: I have seen , my pendant me of spies , Lestrade will be able to in the who asked .
--------------------------------------------------------------------------------
Processing Epoch 14: 100%|██████████| 7611/7611 [19:03<00:00,  6.66it/s, loss_acc=1.743, loss=1.720, lr=6.627383461527933e-05]
--------------------------------------------------------------------------------
    SOURCE: "By God! I tell you you shall drink a glass in here; I'll break the jaws of the first man who looks askance at me!"
    TARGET: —Nom de Dieu! je te dis que tu vas boire une chope la-dedans, je casse la gueule au premier qui me regarde de travers!
 PREDICTED: — Nom de Dieu ! je te dis que tu vas boire un verre , moi , je les mâchoires de l ' air qui me de travers .
--------------------------------------------------------------------------------
    SOURCE: Perhaps other creeks also ran towards the west, but they could not be seen.
    TARGET: Peut-être d'autres creeks couraient-ils vers l'ouest, mais rien ne permettait de le constater.
 PREDICTED: Peut - etre aussi , étant - elle aussi ah ! mais ils ne pouvaient voir de l ' ouest .
--------------------------------------------------------------------------------
Processing Epoch 15:  66%|██████▌   | 5004/7611 [12:10<06:26,  6.74it/s, loss_acc=1.715, loss=1.729, lr=6.38088303725399e-05] 

```

Couldnt complete training afterwards due to colab limits.
