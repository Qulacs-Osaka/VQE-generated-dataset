OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-3.044802273739958) q[0];
ry(0.9287740036182859) q[1];
cx q[0],q[1];
ry(1.9340245743125033) q[0];
ry(-0.49243005900765785) q[1];
cx q[0],q[1];
ry(-2.495084533861125) q[0];
ry(2.469757087144755) q[2];
cx q[0],q[2];
ry(2.683450862153334) q[0];
ry(-2.929454583727086) q[2];
cx q[0],q[2];
ry(-2.539367453573621) q[0];
ry(1.7699106324201752) q[3];
cx q[0],q[3];
ry(0.8718479678389295) q[0];
ry(-1.5779150590849838) q[3];
cx q[0],q[3];
ry(0.7317187758384307) q[1];
ry(-0.8075850932931181) q[2];
cx q[1],q[2];
ry(-2.525544835608259) q[1];
ry(2.283878895859515) q[2];
cx q[1],q[2];
ry(-3.061623208981931) q[1];
ry(-2.2072182082029332) q[3];
cx q[1],q[3];
ry(-1.039089434359367) q[1];
ry(-1.7309121793709323) q[3];
cx q[1],q[3];
ry(-1.9921924363469934) q[2];
ry(1.5875122157002715) q[3];
cx q[2],q[3];
ry(-1.96661358348684) q[2];
ry(0.1484912178193683) q[3];
cx q[2],q[3];
ry(2.4274927005429814) q[0];
ry(-0.3860599306539298) q[1];
cx q[0],q[1];
ry(0.8430333378697096) q[0];
ry(1.7572737892937564) q[1];
cx q[0],q[1];
ry(-2.1759124116698794) q[0];
ry(-0.3862582061524078) q[2];
cx q[0],q[2];
ry(-1.5327167113741782) q[0];
ry(3.0422031632822395) q[2];
cx q[0],q[2];
ry(0.28636867489790363) q[0];
ry(-2.9393454385851356) q[3];
cx q[0],q[3];
ry(-2.6654257962323435) q[0];
ry(-2.131341955459808) q[3];
cx q[0],q[3];
ry(-1.817096707463577) q[1];
ry(2.1821095353591033) q[2];
cx q[1],q[2];
ry(-2.908745193168382) q[1];
ry(1.3862871933208307) q[2];
cx q[1],q[2];
ry(-2.3278467947505255) q[1];
ry(1.1975617367051001) q[3];
cx q[1],q[3];
ry(-0.3665585459591245) q[1];
ry(-2.7783097855888697) q[3];
cx q[1],q[3];
ry(-2.862460490141635) q[2];
ry(2.666325020927834) q[3];
cx q[2],q[3];
ry(0.8687455775762505) q[2];
ry(0.9606586927783445) q[3];
cx q[2],q[3];
ry(1.527061818978062) q[0];
ry(3.035547406370417) q[1];
cx q[0],q[1];
ry(-1.3253875008349834) q[0];
ry(-2.1247163328776484) q[1];
cx q[0],q[1];
ry(2.418113044917911) q[0];
ry(1.4225016857612807) q[2];
cx q[0],q[2];
ry(-2.331242090328345) q[0];
ry(-2.431285882862656) q[2];
cx q[0],q[2];
ry(-1.3269803407026304) q[0];
ry(-1.8946343395473724) q[3];
cx q[0],q[3];
ry(0.5913833665992482) q[0];
ry(-1.9214536273059197) q[3];
cx q[0],q[3];
ry(-2.8148233517187107) q[1];
ry(-2.092782560849958) q[2];
cx q[1],q[2];
ry(-1.3080280830337188) q[1];
ry(-0.31890088156107926) q[2];
cx q[1],q[2];
ry(-2.7213385276301842) q[1];
ry(0.3357035341497588) q[3];
cx q[1],q[3];
ry(-0.41882540239904564) q[1];
ry(2.744944814847091) q[3];
cx q[1],q[3];
ry(-0.003005749545753426) q[2];
ry(-0.3628484525202742) q[3];
cx q[2],q[3];
ry(2.7584175970539646) q[2];
ry(0.7413748446022028) q[3];
cx q[2],q[3];
ry(0.022279788673217848) q[0];
ry(-1.9838912898227967) q[1];
cx q[0],q[1];
ry(0.44665001413782157) q[0];
ry(1.1405342289593448) q[1];
cx q[0],q[1];
ry(3.028838613603737) q[0];
ry(-0.8570589129468542) q[2];
cx q[0],q[2];
ry(2.4811343001983137) q[0];
ry(-1.779593216220458) q[2];
cx q[0],q[2];
ry(-1.6464707484127663) q[0];
ry(1.869959273238635) q[3];
cx q[0],q[3];
ry(-2.048885370804734) q[0];
ry(0.6222560274760829) q[3];
cx q[0],q[3];
ry(-2.534114394992624) q[1];
ry(0.8649203661279877) q[2];
cx q[1],q[2];
ry(0.14081025202551878) q[1];
ry(0.8098305130245854) q[2];
cx q[1],q[2];
ry(1.3320203118470102) q[1];
ry(2.063406122868181) q[3];
cx q[1],q[3];
ry(0.8900424965390723) q[1];
ry(2.671248963278619) q[3];
cx q[1],q[3];
ry(-0.03892735467183221) q[2];
ry(2.116064903815867) q[3];
cx q[2],q[3];
ry(-2.6320915617331764) q[2];
ry(-0.7892263469218959) q[3];
cx q[2],q[3];
ry(1.8549788809334937) q[0];
ry(-1.084307936736248) q[1];
cx q[0],q[1];
ry(1.7128495289936074) q[0];
ry(-0.09593389047355007) q[1];
cx q[0],q[1];
ry(-2.6829700362393076) q[0];
ry(-0.6591715584047234) q[2];
cx q[0],q[2];
ry(-0.9936102167352097) q[0];
ry(0.5702994976403996) q[2];
cx q[0],q[2];
ry(-2.2427001820321486) q[0];
ry(1.2571648841021472) q[3];
cx q[0],q[3];
ry(-1.8717244251681826) q[0];
ry(-2.905011778480443) q[3];
cx q[0],q[3];
ry(-2.961557819183679) q[1];
ry(2.082663917249852) q[2];
cx q[1],q[2];
ry(0.9482757193767937) q[1];
ry(0.3962039589583979) q[2];
cx q[1],q[2];
ry(2.435737173996939) q[1];
ry(1.657521162421742) q[3];
cx q[1],q[3];
ry(1.938020015310645) q[1];
ry(2.438884309291043) q[3];
cx q[1],q[3];
ry(0.5209026865406826) q[2];
ry(-2.92233486249221) q[3];
cx q[2],q[3];
ry(-0.628435532839088) q[2];
ry(-3.1335744071551277) q[3];
cx q[2],q[3];
ry(-2.287076195377489) q[0];
ry(2.3092088696502886) q[1];
cx q[0],q[1];
ry(1.6497124111575603) q[0];
ry(1.8588472373441771) q[1];
cx q[0],q[1];
ry(-2.8860804755091087) q[0];
ry(-1.6641002440822623) q[2];
cx q[0],q[2];
ry(-0.14405952382078532) q[0];
ry(1.8786192060405578) q[2];
cx q[0],q[2];
ry(2.4856302085409343) q[0];
ry(3.077135279816533) q[3];
cx q[0],q[3];
ry(-1.1774671011717812) q[0];
ry(-0.5525920762981924) q[3];
cx q[0],q[3];
ry(-2.523955644020387) q[1];
ry(-1.2530500169445906) q[2];
cx q[1],q[2];
ry(-0.26260097034714364) q[1];
ry(0.116026720972418) q[2];
cx q[1],q[2];
ry(1.177098077544163) q[1];
ry(-2.9222563213161705) q[3];
cx q[1],q[3];
ry(2.7000173036735444) q[1];
ry(0.20178393770703537) q[3];
cx q[1],q[3];
ry(1.6413664594916513) q[2];
ry(-2.6277538476295526) q[3];
cx q[2],q[3];
ry(-2.7418706334792518) q[2];
ry(2.1429260556117105) q[3];
cx q[2],q[3];
ry(-0.11441260446846287) q[0];
ry(-2.021854283686845) q[1];
cx q[0],q[1];
ry(-1.5049614278118728) q[0];
ry(1.7240276290542953) q[1];
cx q[0],q[1];
ry(-0.7979444058550067) q[0];
ry(-2.323890029007009) q[2];
cx q[0],q[2];
ry(-3.017353070772888) q[0];
ry(-1.1927955973002524) q[2];
cx q[0],q[2];
ry(0.2578748694970365) q[0];
ry(2.280662682796576) q[3];
cx q[0],q[3];
ry(-2.40228034741966) q[0];
ry(1.8295437957355478) q[3];
cx q[0],q[3];
ry(-2.697442825910719) q[1];
ry(-1.4743801324228674) q[2];
cx q[1],q[2];
ry(-2.9425047816015883) q[1];
ry(2.8449581474913805) q[2];
cx q[1],q[2];
ry(1.0898615179633353) q[1];
ry(-0.42063028659889784) q[3];
cx q[1],q[3];
ry(-1.8245090871306722) q[1];
ry(-1.6302256775499404) q[3];
cx q[1],q[3];
ry(-1.6815328508033514) q[2];
ry(2.2033575083586783) q[3];
cx q[2],q[3];
ry(1.7184682315656827) q[2];
ry(0.7637594648297599) q[3];
cx q[2],q[3];
ry(-2.5801541565690305) q[0];
ry(-2.145141679781827) q[1];
cx q[0],q[1];
ry(0.7201856645772811) q[0];
ry(-3.1122368878781153) q[1];
cx q[0],q[1];
ry(0.4921016511558882) q[0];
ry(-0.439923664896214) q[2];
cx q[0],q[2];
ry(-2.89151767668611) q[0];
ry(2.1661401584527193) q[2];
cx q[0],q[2];
ry(-2.0102354131434392) q[0];
ry(-1.2904909321425437) q[3];
cx q[0],q[3];
ry(-2.2511390457268865) q[0];
ry(-2.553963136616933) q[3];
cx q[0],q[3];
ry(3.105701737135708) q[1];
ry(-1.5781202395579632) q[2];
cx q[1],q[2];
ry(0.05164243577744961) q[1];
ry(-1.4185269044201956) q[2];
cx q[1],q[2];
ry(0.6068909208650872) q[1];
ry(-2.533194588155499) q[3];
cx q[1],q[3];
ry(-1.2276720366527338) q[1];
ry(-1.9563470206898372) q[3];
cx q[1],q[3];
ry(-0.38564344355189584) q[2];
ry(-1.4621312380391132) q[3];
cx q[2],q[3];
ry(-0.12067717415684094) q[2];
ry(1.4874099864327506) q[3];
cx q[2],q[3];
ry(-2.1179170616470273) q[0];
ry(-2.53293876434381) q[1];
cx q[0],q[1];
ry(-1.8569438989272982) q[0];
ry(-0.7802956309010836) q[1];
cx q[0],q[1];
ry(0.43893610604167677) q[0];
ry(2.0491024403984617) q[2];
cx q[0],q[2];
ry(-1.1030992631075724) q[0];
ry(-2.347325677095238) q[2];
cx q[0],q[2];
ry(-2.0781606812160844) q[0];
ry(2.533176716431012) q[3];
cx q[0],q[3];
ry(-1.3307353328702431) q[0];
ry(1.188020172605397) q[3];
cx q[0],q[3];
ry(0.4940647929527298) q[1];
ry(0.041932141805760714) q[2];
cx q[1],q[2];
ry(-1.3395042406734126) q[1];
ry(3.020558285739811) q[2];
cx q[1],q[2];
ry(-0.9396145492218592) q[1];
ry(-2.261479394773854) q[3];
cx q[1],q[3];
ry(2.1132200117483673) q[1];
ry(-0.5011311754043168) q[3];
cx q[1],q[3];
ry(-2.065220464877193) q[2];
ry(0.27496965301954734) q[3];
cx q[2],q[3];
ry(2.6707287032650404) q[2];
ry(-0.5910056933676211) q[3];
cx q[2],q[3];
ry(-3.0334794886592036) q[0];
ry(2.2184305908280866) q[1];
cx q[0],q[1];
ry(-2.562109929361098) q[0];
ry(1.1682486523815059) q[1];
cx q[0],q[1];
ry(-1.1858674891737362) q[0];
ry(0.26537671746954405) q[2];
cx q[0],q[2];
ry(-1.4966661791996516) q[0];
ry(2.0693946971604413) q[2];
cx q[0],q[2];
ry(-1.519420547384037) q[0];
ry(-1.1780791303388343) q[3];
cx q[0],q[3];
ry(0.667199384258124) q[0];
ry(-2.834416359280275) q[3];
cx q[0],q[3];
ry(-1.399622818508275) q[1];
ry(-2.84299729685563) q[2];
cx q[1],q[2];
ry(1.2549388138782032) q[1];
ry(0.5445325388532536) q[2];
cx q[1],q[2];
ry(1.1775161275180188) q[1];
ry(-2.34461225408836) q[3];
cx q[1],q[3];
ry(-2.727666028216171) q[1];
ry(0.604324320011246) q[3];
cx q[1],q[3];
ry(2.97894290769489) q[2];
ry(-1.7201087064217344) q[3];
cx q[2],q[3];
ry(-1.8835063960060399) q[2];
ry(-0.04522037122042466) q[3];
cx q[2],q[3];
ry(-1.0022037554407284) q[0];
ry(-2.8373630297626646) q[1];
cx q[0],q[1];
ry(-0.3897680811110149) q[0];
ry(-0.5125345165743759) q[1];
cx q[0],q[1];
ry(-2.700651391559219) q[0];
ry(0.015569489526942702) q[2];
cx q[0],q[2];
ry(-2.2678768933472986) q[0];
ry(-0.13189018805249286) q[2];
cx q[0],q[2];
ry(2.151595954833759) q[0];
ry(-1.2618063858680482) q[3];
cx q[0],q[3];
ry(-1.8398586972306699) q[0];
ry(-2.581427819933575) q[3];
cx q[0],q[3];
ry(2.563752628627127) q[1];
ry(2.1028506217282468) q[2];
cx q[1],q[2];
ry(1.9287001915578994) q[1];
ry(-1.8189893040058092) q[2];
cx q[1],q[2];
ry(0.7156544131954031) q[1];
ry(2.103420346092177) q[3];
cx q[1],q[3];
ry(2.1904360845532214) q[1];
ry(-0.9885251388464594) q[3];
cx q[1],q[3];
ry(1.7728150039541382) q[2];
ry(-0.6784075995788337) q[3];
cx q[2],q[3];
ry(2.165819099519275) q[2];
ry(-2.0311816550873227) q[3];
cx q[2],q[3];
ry(-2.9802254442943044) q[0];
ry(1.3861502371876229) q[1];
cx q[0],q[1];
ry(1.078312399871785) q[0];
ry(2.737462452565726) q[1];
cx q[0],q[1];
ry(-1.482118513227556) q[0];
ry(-0.027034078212632165) q[2];
cx q[0],q[2];
ry(-2.9059636149321664) q[0];
ry(0.9123846684171273) q[2];
cx q[0],q[2];
ry(-1.6700321243956848) q[0];
ry(-2.973329220643546) q[3];
cx q[0],q[3];
ry(1.990798597975118) q[0];
ry(1.3539768697620849) q[3];
cx q[0],q[3];
ry(-1.4043114769606273) q[1];
ry(-2.230304632327289) q[2];
cx q[1],q[2];
ry(1.1105911130066715) q[1];
ry(-2.9013873430735315) q[2];
cx q[1],q[2];
ry(2.8431233209137003) q[1];
ry(-0.5002267595535989) q[3];
cx q[1],q[3];
ry(-2.6176713694645377) q[1];
ry(-1.5772092233813364) q[3];
cx q[1],q[3];
ry(3.1107951332354036) q[2];
ry(-2.283827863063756) q[3];
cx q[2],q[3];
ry(-2.8506597641506755) q[2];
ry(-0.678030602303556) q[3];
cx q[2],q[3];
ry(-0.8672920821085743) q[0];
ry(-2.8454775107247134) q[1];
cx q[0],q[1];
ry(-1.7203124740177183) q[0];
ry(0.5879033014024598) q[1];
cx q[0],q[1];
ry(1.7947432505803294) q[0];
ry(-2.765416953701704) q[2];
cx q[0],q[2];
ry(-1.7021507089191834) q[0];
ry(2.687092909645108) q[2];
cx q[0],q[2];
ry(1.507911044384533) q[0];
ry(2.121235965202093) q[3];
cx q[0],q[3];
ry(-2.824367703636275) q[0];
ry(1.9966025479395952) q[3];
cx q[0],q[3];
ry(-1.3797397885353426) q[1];
ry(-1.0003297208703081) q[2];
cx q[1],q[2];
ry(-0.9400818440438901) q[1];
ry(-1.7659707921519043) q[2];
cx q[1],q[2];
ry(0.2616745332179466) q[1];
ry(0.5951610159529119) q[3];
cx q[1],q[3];
ry(-0.03556233910072771) q[1];
ry(0.6555848506748676) q[3];
cx q[1],q[3];
ry(-1.729525780334754) q[2];
ry(-2.300711325117721) q[3];
cx q[2],q[3];
ry(2.5529138568788934) q[2];
ry(-3.012519857825496) q[3];
cx q[2],q[3];
ry(0.2312110821593265) q[0];
ry(0.9576612179272948) q[1];
cx q[0],q[1];
ry(-2.0318385203321316) q[0];
ry(-1.9597581644911966) q[1];
cx q[0],q[1];
ry(-2.592096840794616) q[0];
ry(-0.40154720947008726) q[2];
cx q[0],q[2];
ry(0.5784463640967913) q[0];
ry(2.8983732168331584) q[2];
cx q[0],q[2];
ry(0.13846163747219276) q[0];
ry(-1.4101823427574045) q[3];
cx q[0],q[3];
ry(-0.1546184803158246) q[0];
ry(-0.5465072585845459) q[3];
cx q[0],q[3];
ry(1.7141400638457496) q[1];
ry(-1.901530516769924) q[2];
cx q[1],q[2];
ry(-0.5311825660974376) q[1];
ry(0.5577019874067561) q[2];
cx q[1],q[2];
ry(2.8634185808489994) q[1];
ry(-2.1725003676976784) q[3];
cx q[1],q[3];
ry(-2.8869914887160286) q[1];
ry(-1.0442820918427724) q[3];
cx q[1],q[3];
ry(0.03785428611887465) q[2];
ry(1.7426217121956522) q[3];
cx q[2],q[3];
ry(0.26873852786363206) q[2];
ry(0.6631862513301404) q[3];
cx q[2],q[3];
ry(-0.05268360124322502) q[0];
ry(-3.120208954712344) q[1];
cx q[0],q[1];
ry(3.0078919945065357) q[0];
ry(3.0937012726551023) q[1];
cx q[0],q[1];
ry(2.157849262415729) q[0];
ry(0.11284578630666757) q[2];
cx q[0],q[2];
ry(-1.6449945089220224) q[0];
ry(0.9701214394222183) q[2];
cx q[0],q[2];
ry(2.320218653194015) q[0];
ry(-0.45110935415889214) q[3];
cx q[0],q[3];
ry(-1.9467520724520133) q[0];
ry(0.26211508263262806) q[3];
cx q[0],q[3];
ry(1.373189699846609) q[1];
ry(2.647320326842615) q[2];
cx q[1],q[2];
ry(2.517145788568153) q[1];
ry(-2.1537154115412345) q[2];
cx q[1],q[2];
ry(0.30960312651844046) q[1];
ry(-1.106138865928769) q[3];
cx q[1],q[3];
ry(-0.02714612125060067) q[1];
ry(-0.921997289901217) q[3];
cx q[1],q[3];
ry(2.070572198679416) q[2];
ry(-1.741814414113785) q[3];
cx q[2],q[3];
ry(3.1329781722356707) q[2];
ry(2.7918904021562) q[3];
cx q[2],q[3];
ry(1.3603954411861332) q[0];
ry(-1.1191287693819345) q[1];
cx q[0],q[1];
ry(-2.622523534054404) q[0];
ry(-0.4929268444893759) q[1];
cx q[0],q[1];
ry(2.3308963740011577) q[0];
ry(-0.9168976472548007) q[2];
cx q[0],q[2];
ry(-0.9472655513290242) q[0];
ry(2.0777371585012334) q[2];
cx q[0],q[2];
ry(2.395328990252474) q[0];
ry(2.8324653432910085) q[3];
cx q[0],q[3];
ry(0.32228903776863316) q[0];
ry(-0.1537372399472001) q[3];
cx q[0],q[3];
ry(-3.031144101062997) q[1];
ry(0.18676332665659157) q[2];
cx q[1],q[2];
ry(0.3623378070142334) q[1];
ry(2.3638063128235) q[2];
cx q[1],q[2];
ry(0.7445228392577175) q[1];
ry(-1.3378535137221605) q[3];
cx q[1],q[3];
ry(-2.293457648322377) q[1];
ry(-2.447254591530353) q[3];
cx q[1],q[3];
ry(-0.5922551992725893) q[2];
ry(-2.3843987655798893) q[3];
cx q[2],q[3];
ry(-0.1279125322887653) q[2];
ry(-0.9454720260759323) q[3];
cx q[2],q[3];
ry(1.1961639474139534) q[0];
ry(1.4009567869473518) q[1];
cx q[0],q[1];
ry(3.0227673583132257) q[0];
ry(2.1290700970713052) q[1];
cx q[0],q[1];
ry(0.6346423463004475) q[0];
ry(2.578069399266817) q[2];
cx q[0],q[2];
ry(-0.4053460341859554) q[0];
ry(-1.5898752495757433) q[2];
cx q[0],q[2];
ry(-2.1684432475627062) q[0];
ry(-0.2554194550705615) q[3];
cx q[0],q[3];
ry(0.9335821617860711) q[0];
ry(-0.4824082398863297) q[3];
cx q[0],q[3];
ry(1.6781822691732495) q[1];
ry(-3.011439044256702) q[2];
cx q[1],q[2];
ry(-1.2044391705359097) q[1];
ry(2.3013240591160353) q[2];
cx q[1],q[2];
ry(-1.3058419118435225) q[1];
ry(2.9740113011958833) q[3];
cx q[1],q[3];
ry(0.8958398818564138) q[1];
ry(-1.4533329465023668) q[3];
cx q[1],q[3];
ry(-0.7894896342682732) q[2];
ry(1.882533522763687) q[3];
cx q[2],q[3];
ry(3.0946352242616197) q[2];
ry(-2.022915502823116) q[3];
cx q[2],q[3];
ry(0.8899301239548644) q[0];
ry(-2.25023580061269) q[1];
cx q[0],q[1];
ry(-0.3632044428460288) q[0];
ry(2.809773463629559) q[1];
cx q[0],q[1];
ry(1.7175645156730326) q[0];
ry(-0.6001907282524241) q[2];
cx q[0],q[2];
ry(-2.373717861226273) q[0];
ry(1.2334977533209548) q[2];
cx q[0],q[2];
ry(1.3197629689330759) q[0];
ry(-2.8410283620097063) q[3];
cx q[0],q[3];
ry(-2.292202159801604) q[0];
ry(-1.3219389397289196) q[3];
cx q[0],q[3];
ry(1.288805760525354) q[1];
ry(0.5188165382430457) q[2];
cx q[1],q[2];
ry(1.6678326156324372) q[1];
ry(-1.944393307160087) q[2];
cx q[1],q[2];
ry(-0.16189778964402637) q[1];
ry(-2.111249715172079) q[3];
cx q[1],q[3];
ry(-1.633226746804663) q[1];
ry(2.785848188685861) q[3];
cx q[1],q[3];
ry(-2.8860542868172407) q[2];
ry(0.28091606268265146) q[3];
cx q[2],q[3];
ry(0.44891166942910704) q[2];
ry(-1.569805832159651) q[3];
cx q[2],q[3];
ry(-0.2747491532531381) q[0];
ry(-2.5713434396577326) q[1];
cx q[0],q[1];
ry(1.0712493307529654) q[0];
ry(2.110245605793321) q[1];
cx q[0],q[1];
ry(-0.5157044199529731) q[0];
ry(2.381218485043489) q[2];
cx q[0],q[2];
ry(-1.5724027619646763) q[0];
ry(0.9785993737489518) q[2];
cx q[0],q[2];
ry(2.6922333798778233) q[0];
ry(-0.6098653696286007) q[3];
cx q[0],q[3];
ry(-3.018298005970722) q[0];
ry(2.0815934862229986) q[3];
cx q[0],q[3];
ry(1.851914387037385) q[1];
ry(-0.6846485100059776) q[2];
cx q[1],q[2];
ry(1.8749185222661229) q[1];
ry(-2.3787960737068037) q[2];
cx q[1],q[2];
ry(2.1708824366470525) q[1];
ry(-2.3314772225080884) q[3];
cx q[1],q[3];
ry(-0.8150324581406033) q[1];
ry(0.11411129088572095) q[3];
cx q[1],q[3];
ry(2.717509330423236) q[2];
ry(-2.6354117057412974) q[3];
cx q[2],q[3];
ry(0.14807440458212362) q[2];
ry(1.802419435947317) q[3];
cx q[2],q[3];
ry(-0.630196772302738) q[0];
ry(-2.7408144580397646) q[1];
ry(2.2524752699143) q[2];
ry(-0.39324431409257543) q[3];