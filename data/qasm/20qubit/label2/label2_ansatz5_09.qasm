OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.7389185288235467) q[0];
ry(-2.7265488956001134) q[1];
cx q[0],q[1];
ry(-0.714788869951847) q[0];
ry(-0.9429753647701897) q[1];
cx q[0],q[1];
ry(-0.5794692560196716) q[2];
ry(-2.443239215381441) q[3];
cx q[2],q[3];
ry(1.604304628196214) q[2];
ry(2.103194881292728) q[3];
cx q[2],q[3];
ry(2.909767262208645) q[4];
ry(-0.26151577046406516) q[5];
cx q[4],q[5];
ry(0.7121888082950862) q[4];
ry(-2.0629632306550825) q[5];
cx q[4],q[5];
ry(-0.21171424563286045) q[6];
ry(-1.5647588692002317) q[7];
cx q[6],q[7];
ry(-0.21317216489208488) q[6];
ry(-1.0235309512524169) q[7];
cx q[6],q[7];
ry(0.7017227186941648) q[8];
ry(-0.12334725627158427) q[9];
cx q[8],q[9];
ry(0.6641827666303195) q[8];
ry(0.9342720645592105) q[9];
cx q[8],q[9];
ry(-2.8387694885300854) q[10];
ry(-3.0476233775749284) q[11];
cx q[10],q[11];
ry(2.737182461252979) q[10];
ry(2.1838767476047494) q[11];
cx q[10],q[11];
ry(-2.1529163093578294) q[12];
ry(-0.699682924980845) q[13];
cx q[12],q[13];
ry(-0.5257240882009337) q[12];
ry(-2.742700404651849) q[13];
cx q[12],q[13];
ry(-2.01693013233826) q[14];
ry(2.668514493262962) q[15];
cx q[14],q[15];
ry(1.6510538474372547) q[14];
ry(-2.728903270456331) q[15];
cx q[14],q[15];
ry(-1.1423715054889323) q[16];
ry(-2.933371978637575) q[17];
cx q[16],q[17];
ry(-0.7188921337074232) q[16];
ry(-1.4043374864289042) q[17];
cx q[16],q[17];
ry(0.8926799857281349) q[18];
ry(-0.35447462004615815) q[19];
cx q[18],q[19];
ry(-0.6626681918941187) q[18];
ry(-1.8128311335798986) q[19];
cx q[18],q[19];
ry(0.8060682250150482) q[1];
ry(0.836396724798581) q[2];
cx q[1],q[2];
ry(2.683956923330338) q[1];
ry(-0.8330156037625186) q[2];
cx q[1],q[2];
ry(-1.5015808843010041) q[3];
ry(2.696299952269274) q[4];
cx q[3],q[4];
ry(-3.064589449787879) q[3];
ry(-2.9056685308926293) q[4];
cx q[3],q[4];
ry(-2.5051926656662546) q[5];
ry(-1.865622788903471) q[6];
cx q[5],q[6];
ry(-0.8292859126629103) q[5];
ry(1.0061923609987478) q[6];
cx q[5],q[6];
ry(-0.5393819932867716) q[7];
ry(0.6702842405837739) q[8];
cx q[7],q[8];
ry(3.0825744332138005) q[7];
ry(-1.1113531808938875) q[8];
cx q[7],q[8];
ry(0.15452309532600547) q[9];
ry(0.25923113019270705) q[10];
cx q[9],q[10];
ry(3.0790738114796588) q[9];
ry(1.3424351549278084) q[10];
cx q[9],q[10];
ry(-3.013926134391344) q[11];
ry(2.9730510817713833) q[12];
cx q[11],q[12];
ry(2.584368206440255) q[11];
ry(0.05702529717400357) q[12];
cx q[11],q[12];
ry(-2.301091173341505) q[13];
ry(-2.615690718398901) q[14];
cx q[13],q[14];
ry(-0.0009650389123790617) q[13];
ry(-0.1429303619482258) q[14];
cx q[13],q[14];
ry(-2.5103313485618486) q[15];
ry(-0.5637560524604455) q[16];
cx q[15],q[16];
ry(0.0006647665389724741) q[15];
ry(0.00027662266379385747) q[16];
cx q[15],q[16];
ry(-2.572538416246891) q[17];
ry(-2.2985989753646074) q[18];
cx q[17],q[18];
ry(-0.00022029193122641288) q[17];
ry(-0.0010154957810923904) q[18];
cx q[17],q[18];
ry(-1.8318713780182465) q[0];
ry(-1.2413824944286254) q[1];
cx q[0],q[1];
ry(-0.1007779476982518) q[0];
ry(1.4830240031789232) q[1];
cx q[0],q[1];
ry(-0.17989253343467265) q[2];
ry(3.091091639372537) q[3];
cx q[2],q[3];
ry(-0.1635624705704286) q[2];
ry(0.18067214304656518) q[3];
cx q[2],q[3];
ry(0.5576980130396553) q[4];
ry(-2.142369085981775) q[5];
cx q[4],q[5];
ry(-2.6899318251869793) q[4];
ry(0.0031283687711953874) q[5];
cx q[4],q[5];
ry(2.6324867602587494) q[6];
ry(-0.42622998823466407) q[7];
cx q[6],q[7];
ry(1.5427858926840214) q[6];
ry(1.27981654932446) q[7];
cx q[6],q[7];
ry(-1.6482155795204547) q[8];
ry(-0.5144805568104032) q[9];
cx q[8],q[9];
ry(0.05953275557970806) q[8];
ry(-3.1071087665901187) q[9];
cx q[8],q[9];
ry(2.385154935527355) q[10];
ry(1.8314625799379156) q[11];
cx q[10],q[11];
ry(2.09983892443359) q[10];
ry(0.02757826924179041) q[11];
cx q[10],q[11];
ry(-1.5790427191216831) q[12];
ry(-2.1465842038391365) q[13];
cx q[12],q[13];
ry(-1.7137013760275925) q[12];
ry(-0.589960289825024) q[13];
cx q[12],q[13];
ry(2.833932399847018) q[14];
ry(-0.1423012832739188) q[15];
cx q[14],q[15];
ry(-1.7399119267961438) q[14];
ry(-0.08759549158400315) q[15];
cx q[14],q[15];
ry(-0.4530629508424697) q[16];
ry(-3.134121011290112) q[17];
cx q[16],q[17];
ry(2.0238665230390844) q[16];
ry(2.543513265492051) q[17];
cx q[16],q[17];
ry(2.502321980732057) q[18];
ry(1.4321684414648788) q[19];
cx q[18],q[19];
ry(1.0579604022351345) q[18];
ry(-2.168233639158697) q[19];
cx q[18],q[19];
ry(2.1456927205483023) q[1];
ry(-2.0793279129940996) q[2];
cx q[1],q[2];
ry(-0.8435293680230875) q[1];
ry(1.3107326647840267) q[2];
cx q[1],q[2];
ry(-2.4144334723983856) q[3];
ry(-0.33436426297772054) q[4];
cx q[3],q[4];
ry(0.5127568704239521) q[3];
ry(0.07766394914115625) q[4];
cx q[3],q[4];
ry(-2.4528367682633516) q[5];
ry(-1.1137314866094905) q[6];
cx q[5],q[6];
ry(-2.9191996413067827) q[5];
ry(-0.0638452816987618) q[6];
cx q[5],q[6];
ry(-2.5673893538896864) q[7];
ry(-1.387660494058614) q[8];
cx q[7],q[8];
ry(2.6307467757903615) q[7];
ry(-0.1362097700212747) q[8];
cx q[7],q[8];
ry(3.0056125420124897) q[9];
ry(-1.8871096604368853) q[10];
cx q[9],q[10];
ry(2.723921116038929) q[9];
ry(-2.6120566326867407) q[10];
cx q[9],q[10];
ry(1.5730511227031183) q[11];
ry(-2.8674366636261426) q[12];
cx q[11],q[12];
ry(-1.1296063282739839) q[11];
ry(2.775241474612684) q[12];
cx q[11],q[12];
ry(2.2869767197749185) q[13];
ry(-2.9566080218616335) q[14];
cx q[13],q[14];
ry(0.003735272092319782) q[13];
ry(1.887467992359219) q[14];
cx q[13],q[14];
ry(0.8527763713417036) q[15];
ry(1.8487367003608064) q[16];
cx q[15],q[16];
ry(1.9091173749552595) q[15];
ry(-1.6604109554447) q[16];
cx q[15],q[16];
ry(1.5662144293390243) q[17];
ry(0.6249164265074336) q[18];
cx q[17],q[18];
ry(0.8152260463751914) q[17];
ry(2.179049239079932) q[18];
cx q[17],q[18];
ry(1.1035045397808019) q[0];
ry(2.2465261839106834) q[1];
cx q[0],q[1];
ry(-1.161554646533643) q[0];
ry(0.7168715471525271) q[1];
cx q[0],q[1];
ry(-2.2688325250307404) q[2];
ry(-2.3094922744470154) q[3];
cx q[2],q[3];
ry(0.003240377933641625) q[2];
ry(-0.5333421147169366) q[3];
cx q[2],q[3];
ry(-2.2646380312164003) q[4];
ry(-0.890073357591957) q[5];
cx q[4],q[5];
ry(0.0008521997830284178) q[4];
ry(-3.1395747340560702) q[5];
cx q[4],q[5];
ry(2.4968851112456507) q[6];
ry(2.1339410925068947) q[7];
cx q[6],q[7];
ry(-0.015211299345898155) q[6];
ry(-2.449390638467473) q[7];
cx q[6],q[7];
ry(-2.1073361833463204) q[8];
ry(1.849888857316004) q[9];
cx q[8],q[9];
ry(2.6460842544865475) q[8];
ry(-2.509571803568563) q[9];
cx q[8],q[9];
ry(1.7913109298479037) q[10];
ry(0.2976419020049237) q[11];
cx q[10],q[11];
ry(0.3700498712128591) q[10];
ry(2.4747627730867503) q[11];
cx q[10],q[11];
ry(0.36229790765628184) q[12];
ry(0.9589441940175427) q[13];
cx q[12],q[13];
ry(-0.08589403313546207) q[12];
ry(-0.01956273841545908) q[13];
cx q[12],q[13];
ry(-2.0866059362493337) q[14];
ry(0.2991273366495957) q[15];
cx q[14],q[15];
ry(0.7973104156797204) q[14];
ry(0.01230506511934415) q[15];
cx q[14],q[15];
ry(-1.9140752666132963) q[16];
ry(0.2554501444004813) q[17];
cx q[16],q[17];
ry(3.141065518139001) q[16];
ry(3.141534674563241) q[17];
cx q[16],q[17];
ry(-2.784398079138009) q[18];
ry(0.15260709634977798) q[19];
cx q[18],q[19];
ry(-1.9285425617432095) q[18];
ry(-2.1948160575793256) q[19];
cx q[18],q[19];
ry(-1.8534979564270655) q[1];
ry(2.590212551313999) q[2];
cx q[1],q[2];
ry(-2.015991855479066) q[1];
ry(-1.095019614065968) q[2];
cx q[1],q[2];
ry(-0.19534413244433413) q[3];
ry(-2.638600055381008) q[4];
cx q[3],q[4];
ry(0.1167431277026917) q[3];
ry(-0.9200157948412429) q[4];
cx q[3],q[4];
ry(-2.2481090636936014) q[5];
ry(2.903644223331234) q[6];
cx q[5],q[6];
ry(0.2217432379266843) q[5];
ry(2.979174753527491) q[6];
cx q[5],q[6];
ry(-2.2376798806682503) q[7];
ry(-2.669626114502111) q[8];
cx q[7],q[8];
ry(1.8763128166073784) q[7];
ry(3.1237428967308096) q[8];
cx q[7],q[8];
ry(-1.5896107283454306) q[9];
ry(-0.9074057073674892) q[10];
cx q[9],q[10];
ry(-0.011061125633309169) q[9];
ry(-0.0013677253348712494) q[10];
cx q[9],q[10];
ry(1.2678827427659645) q[11];
ry(-0.2732810719710921) q[12];
cx q[11],q[12];
ry(3.11951580159828) q[11];
ry(2.260770636632607) q[12];
cx q[11],q[12];
ry(2.0244672401982093) q[13];
ry(2.21461350341757) q[14];
cx q[13],q[14];
ry(-3.020249686442623) q[13];
ry(-0.9485410557002236) q[14];
cx q[13],q[14];
ry(2.508961186292678) q[15];
ry(2.842845888713074) q[16];
cx q[15],q[16];
ry(-2.5137899933263954) q[15];
ry(2.1376820021181535) q[16];
cx q[15],q[16];
ry(-0.03646761064899499) q[17];
ry(-0.44700424783175924) q[18];
cx q[17],q[18];
ry(-2.4430185070169497) q[17];
ry(-1.0018943403396166) q[18];
cx q[17],q[18];
ry(1.4494902252609405) q[0];
ry(0.27632845861733446) q[1];
cx q[0],q[1];
ry(0.10304878067798917) q[0];
ry(-0.30217106691496404) q[1];
cx q[0],q[1];
ry(1.660752242592129) q[2];
ry(-2.2104941152078883) q[3];
cx q[2],q[3];
ry(-3.1335021817203095) q[2];
ry(-1.8274748104127703) q[3];
cx q[2],q[3];
ry(-2.2294202724242123) q[4];
ry(-2.311048532478863) q[5];
cx q[4],q[5];
ry(0.0015456669716122917) q[4];
ry(0.001338114196299145) q[5];
cx q[4],q[5];
ry(1.6100657593656624) q[6];
ry(-0.389426167829927) q[7];
cx q[6],q[7];
ry(-0.009176620197114389) q[6];
ry(2.186907276494706) q[7];
cx q[6],q[7];
ry(1.3759698356432377) q[8];
ry(2.086753917173567) q[9];
cx q[8],q[9];
ry(-0.1573975694841634) q[8];
ry(1.6274565253094049) q[9];
cx q[8],q[9];
ry(1.0305314376790875) q[10];
ry(-0.18172982451781078) q[11];
cx q[10],q[11];
ry(-0.4181861974973371) q[10];
ry(-0.4833194981044508) q[11];
cx q[10],q[11];
ry(0.5184987299830826) q[12];
ry(0.653441650939115) q[13];
cx q[12],q[13];
ry(2.9218525568588305) q[12];
ry(3.136926692481785) q[13];
cx q[12],q[13];
ry(2.877766981477329) q[14];
ry(-1.5126299922155213) q[15];
cx q[14],q[15];
ry(-0.311398410934216) q[14];
ry(-0.47790449875378815) q[15];
cx q[14],q[15];
ry(1.7103947386212404) q[16];
ry(2.351272160244681) q[17];
cx q[16],q[17];
ry(0.006174769507369505) q[16];
ry(-3.035687336866931) q[17];
cx q[16],q[17];
ry(1.2466850959757299) q[18];
ry(-0.9563646606152791) q[19];
cx q[18],q[19];
ry(-2.2873091334010294) q[18];
ry(2.618705476715649) q[19];
cx q[18],q[19];
ry(2.675051811213351) q[1];
ry(-1.8531906611083135) q[2];
cx q[1],q[2];
ry(1.3532501732319404) q[1];
ry(1.0419121390939035) q[2];
cx q[1],q[2];
ry(-2.7941503315937544) q[3];
ry(-0.4409470608812854) q[4];
cx q[3],q[4];
ry(0.3984166347810093) q[3];
ry(-2.3372790682020788) q[4];
cx q[3],q[4];
ry(-1.4356446880589688) q[5];
ry(-2.460277145751727) q[6];
cx q[5],q[6];
ry(2.912797635270847) q[5];
ry(-0.076385927098368) q[6];
cx q[5],q[6];
ry(-3.0430049346212407) q[7];
ry(-1.3957413293203906) q[8];
cx q[7],q[8];
ry(-1.551207024157086) q[7];
ry(0.030795006530581755) q[8];
cx q[7],q[8];
ry(-0.4401531568811623) q[9];
ry(2.609790535948816) q[10];
cx q[9],q[10];
ry(-0.0035550342251400393) q[9];
ry(3.119681652263949) q[10];
cx q[9],q[10];
ry(2.318829922353849) q[11];
ry(1.0691275866558714) q[12];
cx q[11],q[12];
ry(-3.1108583368704803) q[11];
ry(0.6989775067924522) q[12];
cx q[11],q[12];
ry(3.0382724144858195) q[13];
ry(2.316286946090724) q[14];
cx q[13],q[14];
ry(-1.0054887713286682) q[13];
ry(2.5558054259741665) q[14];
cx q[13],q[14];
ry(-2.449275369047272) q[15];
ry(2.6257724405124976) q[16];
cx q[15],q[16];
ry(1.4780497923946219) q[15];
ry(3.0272656270294678) q[16];
cx q[15],q[16];
ry(1.3773676599170361) q[17];
ry(-2.6820222246651038) q[18];
cx q[17],q[18];
ry(2.0308901638853873) q[17];
ry(-0.007911623651495472) q[18];
cx q[17],q[18];
ry(0.9038764356534379) q[0];
ry(2.5614939254682625) q[1];
cx q[0],q[1];
ry(-2.083563958503289) q[0];
ry(-2.5389768059775712) q[1];
cx q[0],q[1];
ry(-2.487401324403869) q[2];
ry(0.5356872406074483) q[3];
cx q[2],q[3];
ry(-2.1803780681721054) q[2];
ry(2.3470110413550853) q[3];
cx q[2],q[3];
ry(1.7200821878538508) q[4];
ry(0.34080245059481307) q[5];
cx q[4],q[5];
ry(1.8075971951194374) q[4];
ry(-0.018284985143580797) q[5];
cx q[4],q[5];
ry(-1.3583065759573474) q[6];
ry(-0.10499996957478629) q[7];
cx q[6],q[7];
ry(0.04151345883626977) q[6];
ry(-1.8098554256760409) q[7];
cx q[6],q[7];
ry(-2.1474703164523277) q[8];
ry(-2.6856085238852208) q[9];
cx q[8],q[9];
ry(-2.1627398327504643) q[8];
ry(-1.5215368182763915) q[9];
cx q[8],q[9];
ry(2.594366654516599) q[10];
ry(1.9854431386728522) q[11];
cx q[10],q[11];
ry(0.9432940763225587) q[10];
ry(-0.6154939344358059) q[11];
cx q[10],q[11];
ry(-0.2819904356336256) q[12];
ry(3.0480474318133375) q[13];
cx q[12],q[13];
ry(1.2233386116591918) q[12];
ry(0.006233048885516013) q[13];
cx q[12],q[13];
ry(-0.8974967597304451) q[14];
ry(-1.3298055594538036) q[15];
cx q[14],q[15];
ry(2.7243959444666688) q[14];
ry(2.643861895301112) q[15];
cx q[14],q[15];
ry(2.1450129250522894) q[16];
ry(-1.227609619474906) q[17];
cx q[16],q[17];
ry(-0.41624288044524077) q[16];
ry(-0.12845138517278146) q[17];
cx q[16],q[17];
ry(0.8199125719919086) q[18];
ry(-2.839464263648027) q[19];
cx q[18],q[19];
ry(-1.7404021691751783) q[18];
ry(-1.9832655560119976) q[19];
cx q[18],q[19];
ry(-0.3656333661083908) q[1];
ry(-0.5965777253277812) q[2];
cx q[1],q[2];
ry(-0.503695074110113) q[1];
ry(2.8882987353569867) q[2];
cx q[1],q[2];
ry(1.9075432271391026) q[3];
ry(-0.9433609124621699) q[4];
cx q[3],q[4];
ry(0.005531860611717109) q[3];
ry(0.2391566218961545) q[4];
cx q[3],q[4];
ry(-2.084385680273618) q[5];
ry(-0.38935749942325515) q[6];
cx q[5],q[6];
ry(-2.983322918437916) q[5];
ry(-2.345014974227784) q[6];
cx q[5],q[6];
ry(-2.2182851328679316) q[7];
ry(-2.626920226788193) q[8];
cx q[7],q[8];
ry(0.08116986256936691) q[7];
ry(-0.1517478567157795) q[8];
cx q[7],q[8];
ry(-1.328411058392444) q[9];
ry(-1.154837819672995) q[10];
cx q[9],q[10];
ry(-3.128594714141825) q[9];
ry(3.1341650976207895) q[10];
cx q[9],q[10];
ry(0.6815088517821266) q[11];
ry(-0.6518104440002526) q[12];
cx q[11],q[12];
ry(0.0024881741294864317) q[11];
ry(-1.2226032903481852) q[12];
cx q[11],q[12];
ry(0.9697705326630341) q[13];
ry(-1.3754231152002125) q[14];
cx q[13],q[14];
ry(-0.040972688371989) q[13];
ry(2.543976734766635) q[14];
cx q[13],q[14];
ry(-1.6867142796028611) q[15];
ry(0.7569506141400888) q[16];
cx q[15],q[16];
ry(0.7801798049934137) q[15];
ry(0.23596316068713813) q[16];
cx q[15],q[16];
ry(-2.2653655149151923) q[17];
ry(-0.911947536543829) q[18];
cx q[17],q[18];
ry(-0.28682521824937446) q[17];
ry(3.0681608456769682) q[18];
cx q[17],q[18];
ry(-2.9108507957353655) q[0];
ry(-1.8760029789567758) q[1];
cx q[0],q[1];
ry(-0.43704707818420463) q[0];
ry(2.100698426704958) q[1];
cx q[0],q[1];
ry(3.087320566447454) q[2];
ry(-1.002870543999853) q[3];
cx q[2],q[3];
ry(0.7874262180039557) q[2];
ry(2.5708537553647273) q[3];
cx q[2],q[3];
ry(-1.3104599450416627) q[4];
ry(-1.4754794245238259) q[5];
cx q[4],q[5];
ry(-2.9486708450622316) q[4];
ry(-0.0029089015916786342) q[5];
cx q[4],q[5];
ry(0.18225539544847083) q[6];
ry(-2.092812823802411) q[7];
cx q[6],q[7];
ry(0.16221803927729347) q[6];
ry(-3.062417468289494) q[7];
cx q[6],q[7];
ry(-1.6868172523431748) q[8];
ry(-0.32271079781771184) q[9];
cx q[8],q[9];
ry(0.8182287753025042) q[8];
ry(0.8169067915179511) q[9];
cx q[8],q[9];
ry(2.934039804216283) q[10];
ry(2.7024968491702093) q[11];
cx q[10],q[11];
ry(-2.96387881082027) q[10];
ry(-2.9251360349738067) q[11];
cx q[10],q[11];
ry(2.8270595034140418) q[12];
ry(-2.006742535641028) q[13];
cx q[12],q[13];
ry(-1.2596244330837647) q[12];
ry(-0.006017375798166747) q[13];
cx q[12],q[13];
ry(-0.009716127149274278) q[14];
ry(1.7298595938353138) q[15];
cx q[14],q[15];
ry(-2.8952643642308513) q[14];
ry(-0.0037395472911283534) q[15];
cx q[14],q[15];
ry(1.962189407254966) q[16];
ry(1.1137248294966853) q[17];
cx q[16],q[17];
ry(-0.002789597963881742) q[16];
ry(0.4568728710400194) q[17];
cx q[16],q[17];
ry(-1.2330634967369491) q[18];
ry(2.726701944087512) q[19];
cx q[18],q[19];
ry(0.6332686809527655) q[18];
ry(0.8508523459459587) q[19];
cx q[18],q[19];
ry(1.7911134101889503) q[1];
ry(-0.3906410792673534) q[2];
cx q[1],q[2];
ry(-0.5383220679287373) q[1];
ry(-2.331607366607673) q[2];
cx q[1],q[2];
ry(-0.4884115328957144) q[3];
ry(-2.013257990363323) q[4];
cx q[3],q[4];
ry(3.0827654928943145) q[3];
ry(0.7511955232774322) q[4];
cx q[3],q[4];
ry(-1.4888988876291125) q[5];
ry(-2.8911741778318314) q[6];
cx q[5],q[6];
ry(-2.9751650552247244) q[5];
ry(2.4376847045096786) q[6];
cx q[5],q[6];
ry(2.86588879081328) q[7];
ry(-1.024628438164254) q[8];
cx q[7],q[8];
ry(3.0527207492751867) q[7];
ry(2.912901639422834) q[8];
cx q[7],q[8];
ry(1.9827176193901614) q[9];
ry(3.0255248322243147) q[10];
cx q[9],q[10];
ry(-0.0060560706700254485) q[9];
ry(-0.976924423638535) q[10];
cx q[9],q[10];
ry(-1.985525505663949) q[11];
ry(1.238458468501821) q[12];
cx q[11],q[12];
ry(-0.030227602859580035) q[11];
ry(-0.9854875994903001) q[12];
cx q[11],q[12];
ry(0.136638989200355) q[13];
ry(-0.1777851640038426) q[14];
cx q[13],q[14];
ry(1.2340673962063793) q[13];
ry(1.8972631176155363) q[14];
cx q[13],q[14];
ry(1.1018628675479274) q[15];
ry(-2.6837458818530684) q[16];
cx q[15],q[16];
ry(1.923070651260165) q[15];
ry(2.839092701650016) q[16];
cx q[15],q[16];
ry(1.4750264089749812) q[17];
ry(1.5914230180201825) q[18];
cx q[17],q[18];
ry(0.7733859012598092) q[17];
ry(0.0014837632353028093) q[18];
cx q[17],q[18];
ry(1.6912234512155622) q[0];
ry(2.3494953519145665) q[1];
cx q[0],q[1];
ry(-1.9767356324263101) q[0];
ry(0.5584131622696962) q[1];
cx q[0],q[1];
ry(1.4523429891154978) q[2];
ry(0.703919293727116) q[3];
cx q[2],q[3];
ry(-2.504081849434176) q[2];
ry(-0.005635316406544865) q[3];
cx q[2],q[3];
ry(0.7722937828843338) q[4];
ry(1.8286856173228896) q[5];
cx q[4],q[5];
ry(-2.4512492736240157) q[4];
ry(-0.026325115286380884) q[5];
cx q[4],q[5];
ry(-2.370936722670412) q[6];
ry(0.014262813888155534) q[7];
cx q[6],q[7];
ry(-2.5708735195817605) q[6];
ry(-2.9409174963708065) q[7];
cx q[6],q[7];
ry(-0.17566999607869946) q[8];
ry(1.5952169767578797) q[9];
cx q[8],q[9];
ry(-2.7690532684600444) q[8];
ry(-0.003098538395078919) q[9];
cx q[8],q[9];
ry(-2.2430317244167526) q[10];
ry(-1.3961366355860885) q[11];
cx q[10],q[11];
ry(1.13295557325704) q[10];
ry(3.0290655274483) q[11];
cx q[10],q[11];
ry(0.9569993118429392) q[12];
ry(0.8551674837269534) q[13];
cx q[12],q[13];
ry(0.18394817099268196) q[12];
ry(-0.00038679016844245234) q[13];
cx q[12],q[13];
ry(0.7999267658252984) q[14];
ry(0.9483190600379521) q[15];
cx q[14],q[15];
ry(-0.004372813177340618) q[14];
ry(-3.136612383799586) q[15];
cx q[14],q[15];
ry(-3.0218791527811146) q[16];
ry(-1.9641819553959168) q[17];
cx q[16],q[17];
ry(-3.05932351167099) q[16];
ry(2.9520412070216184) q[17];
cx q[16],q[17];
ry(-0.22501453704066687) q[18];
ry(0.33542053191432136) q[19];
cx q[18],q[19];
ry(-0.6487114525152959) q[18];
ry(-1.644484054927478) q[19];
cx q[18],q[19];
ry(2.0144911571467246) q[1];
ry(1.3986940617194616) q[2];
cx q[1],q[2];
ry(0.9004348171826583) q[1];
ry(1.1938819683054112) q[2];
cx q[1],q[2];
ry(1.237306598619778) q[3];
ry(1.9168978209650813) q[4];
cx q[3],q[4];
ry(-0.016706654218750398) q[3];
ry(-1.3181595411245688) q[4];
cx q[3],q[4];
ry(-2.5174517406977053) q[5];
ry(-2.289739893783085) q[6];
cx q[5],q[6];
ry(3.1260393783117264) q[5];
ry(3.133244351929049) q[6];
cx q[5],q[6];
ry(-2.6188097552065175) q[7];
ry(0.06404904000117462) q[8];
cx q[7],q[8];
ry(0.4937734451678982) q[7];
ry(2.9192419622338495) q[8];
cx q[7],q[8];
ry(-1.5533354241950879) q[9];
ry(0.45043901444939655) q[10];
cx q[9],q[10];
ry(-0.023710038149661983) q[9];
ry(-1.1106502185210376) q[10];
cx q[9],q[10];
ry(-1.7112049902960411) q[11];
ry(-2.4727272152508855) q[12];
cx q[11],q[12];
ry(-0.0022394148855180134) q[11];
ry(0.7916877795577929) q[12];
cx q[11],q[12];
ry(-0.35600219886506596) q[13];
ry(1.8641517227133215) q[14];
cx q[13],q[14];
ry(0.04012937110491954) q[13];
ry(0.4634427149185479) q[14];
cx q[13],q[14];
ry(1.8506366428742007) q[15];
ry(1.214099706153657) q[16];
cx q[15],q[16];
ry(-2.9000001145724132) q[15];
ry(-0.7276435632621308) q[16];
cx q[15],q[16];
ry(-1.1034146306646289) q[17];
ry(2.542402163246479) q[18];
cx q[17],q[18];
ry(2.561188299160895) q[17];
ry(3.0835829675684376) q[18];
cx q[17],q[18];
ry(-1.0950354539304132) q[0];
ry(2.491070165508709) q[1];
cx q[0],q[1];
ry(1.3534282422006694) q[0];
ry(2.2238006717268224) q[1];
cx q[0],q[1];
ry(1.2994041032415717) q[2];
ry(3.030083224169183) q[3];
cx q[2],q[3];
ry(-2.0537802864243426) q[2];
ry(-0.011762670944859766) q[3];
cx q[2],q[3];
ry(-1.3603706633042172) q[4];
ry(1.2963435538860564) q[5];
cx q[4],q[5];
ry(1.0229470300800756) q[4];
ry(3.111837596324257) q[5];
cx q[4],q[5];
ry(1.1328408538496344) q[6];
ry(2.3390451944065225) q[7];
cx q[6],q[7];
ry(0.3236148429563664) q[6];
ry(-3.0348535837766155) q[7];
cx q[6],q[7];
ry(1.2810014618408971) q[8];
ry(0.5280433457188093) q[9];
cx q[8],q[9];
ry(2.0828963815051615) q[8];
ry(1.3842216829551761) q[9];
cx q[8],q[9];
ry(2.9051750537203422) q[10];
ry(1.7199483160876659) q[11];
cx q[10],q[11];
ry(-1.2728026176251246) q[10];
ry(-1.9475255883042293) q[11];
cx q[10],q[11];
ry(1.5012593837104315) q[12];
ry(2.8944739734330134) q[13];
cx q[12],q[13];
ry(-0.07538576836199141) q[12];
ry(-3.139532539437768) q[13];
cx q[12],q[13];
ry(-2.285809440556075) q[14];
ry(2.084559952451965) q[15];
cx q[14],q[15];
ry(-3.118491963232059) q[14];
ry(0.02258100507085779) q[15];
cx q[14],q[15];
ry(-1.5481777019712846) q[16];
ry(1.340259415882203) q[17];
cx q[16],q[17];
ry(3.033831635499618) q[16];
ry(0.6229667778273343) q[17];
cx q[16],q[17];
ry(-0.11556745676838975) q[18];
ry(-1.538159326663982) q[19];
cx q[18],q[19];
ry(-0.3674195455685165) q[18];
ry(-2.3563530432405004) q[19];
cx q[18],q[19];
ry(2.4135121680326304) q[1];
ry(0.7287297968490298) q[2];
cx q[1],q[2];
ry(3.0509226143849144) q[1];
ry(-0.22077376372491744) q[2];
cx q[1],q[2];
ry(-0.2778139786851215) q[3];
ry(2.7177167715297785) q[4];
cx q[3],q[4];
ry(0.009527954115819526) q[3];
ry(0.2506251323245856) q[4];
cx q[3],q[4];
ry(0.8730124751714587) q[5];
ry(0.8726918505353476) q[6];
cx q[5],q[6];
ry(3.111953876036685) q[5];
ry(-0.04146416996270563) q[6];
cx q[5],q[6];
ry(-2.0902444022646565) q[7];
ry(2.6814762084789243) q[8];
cx q[7],q[8];
ry(-1.7323181795689426) q[7];
ry(-0.05593210757131128) q[8];
cx q[7],q[8];
ry(-2.8858608467460827) q[9];
ry(-1.7256179438727495) q[10];
cx q[9],q[10];
ry(0.0007984997401651412) q[9];
ry(-3.1410454424531373) q[10];
cx q[9],q[10];
ry(1.0679251311524958) q[11];
ry(-0.06477988174545551) q[12];
cx q[11],q[12];
ry(-0.049428990134152244) q[11];
ry(2.20210582008987) q[12];
cx q[11],q[12];
ry(-2.496780731859429) q[13];
ry(-1.7272277682324917) q[14];
cx q[13],q[14];
ry(-2.183845448436036) q[13];
ry(1.7490074643186837) q[14];
cx q[13],q[14];
ry(-1.458815855532948) q[15];
ry(-1.386080728685947) q[16];
cx q[15],q[16];
ry(-0.12520056561663306) q[15];
ry(-0.13875788247763393) q[16];
cx q[15],q[16];
ry(1.5829806296256042) q[17];
ry(-1.0933340012019874) q[18];
cx q[17],q[18];
ry(2.4445693090647502) q[17];
ry(-2.9639209753656894) q[18];
cx q[17],q[18];
ry(0.9360126524305112) q[0];
ry(-1.1698862401318193) q[1];
cx q[0],q[1];
ry(-0.9132346216262542) q[0];
ry(-0.5356471514479192) q[1];
cx q[0],q[1];
ry(-2.8398115229350527) q[2];
ry(2.7052749036882955) q[3];
cx q[2],q[3];
ry(-0.6189484571470358) q[2];
ry(1.8050159549295934) q[3];
cx q[2],q[3];
ry(1.7900905090945436) q[4];
ry(-0.9752440054390803) q[5];
cx q[4],q[5];
ry(-2.269679780092948) q[4];
ry(-0.12221533411985863) q[5];
cx q[4],q[5];
ry(-1.1059046224795523) q[6];
ry(3.0333457920476294) q[7];
cx q[6],q[7];
ry(-3.0595573815551895) q[6];
ry(0.0863232879175202) q[7];
cx q[6],q[7];
ry(0.03454876353996639) q[8];
ry(-1.1272231813792253) q[9];
cx q[8],q[9];
ry(-0.5305055644790956) q[8];
ry(-2.776567136240901) q[9];
cx q[8],q[9];
ry(2.585949262249057) q[10];
ry(-1.2371844101644427) q[11];
cx q[10],q[11];
ry(-1.9283450120329695) q[10];
ry(-2.271792112816992) q[11];
cx q[10],q[11];
ry(-3.129668301415183) q[12];
ry(1.0833211838493817) q[13];
cx q[12],q[13];
ry(-1.8563899683457266) q[12];
ry(0.8486038563642333) q[13];
cx q[12],q[13];
ry(-1.7761731535069405) q[14];
ry(-3.1270161318523666) q[15];
cx q[14],q[15];
ry(2.7213348925771546) q[14];
ry(-2.016582763434821) q[15];
cx q[14],q[15];
ry(0.644233348696876) q[16];
ry(0.9957998731301716) q[17];
cx q[16],q[17];
ry(1.331408161644189) q[16];
ry(3.0978220647196526) q[17];
cx q[16],q[17];
ry(-2.071759598847371) q[18];
ry(-1.7582750712962065) q[19];
cx q[18],q[19];
ry(2.8015057976695865) q[18];
ry(0.9366488449567025) q[19];
cx q[18],q[19];
ry(2.876684024993719) q[1];
ry(-1.3283044957828656) q[2];
cx q[1],q[2];
ry(-0.2249442905706207) q[1];
ry(2.8259650796398117) q[2];
cx q[1],q[2];
ry(1.178395826668516) q[3];
ry(-1.8777463029621455) q[4];
cx q[3],q[4];
ry(-3.136927496828807) q[3];
ry(3.110097084174567) q[4];
cx q[3],q[4];
ry(1.1791239816602583) q[5];
ry(-1.8720130283494782) q[6];
cx q[5],q[6];
ry(0.004903796164111185) q[5];
ry(3.1217350830542547) q[6];
cx q[5],q[6];
ry(-0.7886701605302124) q[7];
ry(-0.46419327438847074) q[8];
cx q[7],q[8];
ry(-1.4991149750756696) q[7];
ry(3.051934708411867) q[8];
cx q[7],q[8];
ry(-0.036097884737494645) q[9];
ry(1.4804229514198044) q[10];
cx q[9],q[10];
ry(-3.14063686449142) q[9];
ry(0.0014675304355480116) q[10];
cx q[9],q[10];
ry(1.2919562211500368) q[11];
ry(-1.4930589409423203) q[12];
cx q[11],q[12];
ry(2.2020922810752483) q[11];
ry(-3.1379848518185858) q[12];
cx q[11],q[12];
ry(-2.341000872877543) q[13];
ry(-2.6290408247838424) q[14];
cx q[13],q[14];
ry(-0.012820640513400148) q[13];
ry(0.006009860556918188) q[14];
cx q[13],q[14];
ry(-2.6215810954772807) q[15];
ry(0.010829753179917745) q[16];
cx q[15],q[16];
ry(0.0028606452239858643) q[15];
ry(3.131310413370077) q[16];
cx q[15],q[16];
ry(0.6319768371993826) q[17];
ry(0.7429952305385967) q[18];
cx q[17],q[18];
ry(-0.17753944897867502) q[17];
ry(3.1190115456662415) q[18];
cx q[17],q[18];
ry(0.1824305562464934) q[0];
ry(0.7997991902285216) q[1];
cx q[0],q[1];
ry(-1.3890646187239795) q[0];
ry(-1.2673301417619474) q[1];
cx q[0],q[1];
ry(3.0423931063366663) q[2];
ry(2.9379682012541375) q[3];
cx q[2],q[3];
ry(1.6166809181073711) q[2];
ry(-2.934976891126432) q[3];
cx q[2],q[3];
ry(0.22675760029139796) q[4];
ry(3.1179874744777654) q[5];
cx q[4],q[5];
ry(1.0947280779330655) q[4];
ry(-0.006127296130118865) q[5];
cx q[4],q[5];
ry(2.4637870267423967) q[6];
ry(-1.7076919351821136) q[7];
cx q[6],q[7];
ry(-0.7545322289972285) q[6];
ry(-1.3516978471373102) q[7];
cx q[6],q[7];
ry(-2.233103547650863) q[8];
ry(2.0687975190054226) q[9];
cx q[8],q[9];
ry(1.9052219984697067) q[8];
ry(1.2157253883463215) q[9];
cx q[8],q[9];
ry(0.5308797553738103) q[10];
ry(-1.3945493355101446) q[11];
cx q[10],q[11];
ry(3.1229057384010974) q[10];
ry(-1.3561790601483585) q[11];
cx q[10],q[11];
ry(-1.4281715214377135) q[12];
ry(0.7364680456408147) q[13];
cx q[12],q[13];
ry(0.07058610565363566) q[12];
ry(-0.9868536704512911) q[13];
cx q[12],q[13];
ry(-1.0610367737106197) q[14];
ry(-1.031657947530394) q[15];
cx q[14],q[15];
ry(-2.3443071047085042) q[14];
ry(-0.19472300231910075) q[15];
cx q[14],q[15];
ry(-2.578921424460857) q[16];
ry(0.6416467465848855) q[17];
cx q[16],q[17];
ry(0.9828999320986531) q[16];
ry(-1.6648889365395894) q[17];
cx q[16],q[17];
ry(0.8434839630633943) q[18];
ry(-2.525561369117785) q[19];
cx q[18],q[19];
ry(0.9694498687809077) q[18];
ry(3.100174552366911) q[19];
cx q[18],q[19];
ry(-0.5207702008450665) q[1];
ry(-3.0300313200445377) q[2];
cx q[1],q[2];
ry(0.5792634657520548) q[1];
ry(3.0897174430640075) q[2];
cx q[1],q[2];
ry(1.1725768831839565) q[3];
ry(-0.0649282457780815) q[4];
cx q[3],q[4];
ry(3.0377986458667827) q[3];
ry(-0.08638153686289753) q[4];
cx q[3],q[4];
ry(-0.4460283224693218) q[5];
ry(1.5060819152628238) q[6];
cx q[5],q[6];
ry(3.125300448944166) q[5];
ry(-3.1280999630149995) q[6];
cx q[5],q[6];
ry(-0.14566979863935448) q[7];
ry(-2.179485613443045) q[8];
cx q[7],q[8];
ry(-3.0736887898319343) q[7];
ry(0.9445500883055064) q[8];
cx q[7],q[8];
ry(1.2549937844326033) q[9];
ry(-1.2771237238727444) q[10];
cx q[9],q[10];
ry(-0.03143280031088658) q[9];
ry(1.0535354825245966) q[10];
cx q[9],q[10];
ry(-1.4799561650196615) q[11];
ry(1.5558195799204038) q[12];
cx q[11],q[12];
ry(-0.953019665627893) q[11];
ry(-2.8751733758881395) q[12];
cx q[11],q[12];
ry(2.415011404666114) q[13];
ry(-0.7013652068500402) q[14];
cx q[13],q[14];
ry(0.0023454526355348535) q[13];
ry(-0.06778482777854045) q[14];
cx q[13],q[14];
ry(0.5408237832264877) q[15];
ry(1.2826055854109484) q[16];
cx q[15],q[16];
ry(-0.10297126495445497) q[15];
ry(-0.010913695495290199) q[16];
cx q[15],q[16];
ry(-1.9011992601049867) q[17];
ry(-2.7183466233969353) q[18];
cx q[17],q[18];
ry(-1.4524159017127518) q[17];
ry(-3.1319738165284154) q[18];
cx q[17],q[18];
ry(-2.6370569762681972) q[0];
ry(-0.7985805461649083) q[1];
cx q[0],q[1];
ry(-2.7603954635204007) q[0];
ry(-1.7464166991437542) q[1];
cx q[0],q[1];
ry(0.7912931484616053) q[2];
ry(2.6692185830875794) q[3];
cx q[2],q[3];
ry(0.6398391265660472) q[2];
ry(-1.82259243566123) q[3];
cx q[2],q[3];
ry(-0.8871688202107032) q[4];
ry(0.3719826853499254) q[5];
cx q[4],q[5];
ry(-0.00158018719248233) q[4];
ry(-3.0962404743901866) q[5];
cx q[4],q[5];
ry(-2.035663366273452) q[6];
ry(-2.391196394115028) q[7];
cx q[6],q[7];
ry(-3.07620111852764) q[6];
ry(-3.03017769398679) q[7];
cx q[6],q[7];
ry(-0.9545726100383752) q[8];
ry(1.552000082086156) q[9];
cx q[8],q[9];
ry(0.889433983427465) q[8];
ry(-3.1347249504723798) q[9];
cx q[8],q[9];
ry(-2.7909672254545073) q[10];
ry(2.890026254871058) q[11];
cx q[10],q[11];
ry(-3.1361148744319545) q[10];
ry(-0.001360933614412474) q[11];
cx q[10],q[11];
ry(-1.5876777763009073) q[12];
ry(-1.500660632098242) q[13];
cx q[12],q[13];
ry(-0.6699487011561348) q[12];
ry(-0.06632049518152133) q[13];
cx q[12],q[13];
ry(3.1142365977559026) q[14];
ry(-2.772179844560658) q[15];
cx q[14],q[15];
ry(-1.7153739682788443) q[14];
ry(-0.5900876369214968) q[15];
cx q[14],q[15];
ry(0.7969906858714317) q[16];
ry(-1.4003002497647374) q[17];
cx q[16],q[17];
ry(-3.11510167329315) q[16];
ry(2.547040618315518) q[17];
cx q[16],q[17];
ry(2.563498091287035) q[18];
ry(-1.404150893720644) q[19];
cx q[18],q[19];
ry(-1.0960726976358823) q[18];
ry(-1.2098568057194399) q[19];
cx q[18],q[19];
ry(-0.5851350590666602) q[1];
ry(2.7348958119226245) q[2];
cx q[1],q[2];
ry(-2.787173459024847) q[1];
ry(-1.6856890795254391) q[2];
cx q[1],q[2];
ry(-2.151057666970857) q[3];
ry(0.6054091456761509) q[4];
cx q[3],q[4];
ry(-3.0181523326069413) q[3];
ry(3.0151063320875973) q[4];
cx q[3],q[4];
ry(2.409812265118738) q[5];
ry(2.0199747466213447) q[6];
cx q[5],q[6];
ry(-0.035319142972477624) q[5];
ry(0.029854361893812964) q[6];
cx q[5],q[6];
ry(-2.297251051716406) q[7];
ry(3.005997423834905) q[8];
cx q[7],q[8];
ry(3.0368709209905593) q[7];
ry(0.9802280392089999) q[8];
cx q[7],q[8];
ry(-1.2730762009637875) q[9];
ry(2.751464549641721) q[10];
cx q[9],q[10];
ry(-0.1534180326914738) q[9];
ry(-2.1190124502022485) q[10];
cx q[9],q[10];
ry(0.5835085982790282) q[11];
ry(1.229993427088099) q[12];
cx q[11],q[12];
ry(-0.9812533749249566) q[11];
ry(2.4093684982418235) q[12];
cx q[11],q[12];
ry(-1.8565817031386727) q[13];
ry(0.7767166710080424) q[14];
cx q[13],q[14];
ry(3.1097084468539946) q[13];
ry(3.0899130584857812) q[14];
cx q[13],q[14];
ry(-1.8213979098543858) q[15];
ry(0.11714026099742036) q[16];
cx q[15],q[16];
ry(0.12052082976810308) q[15];
ry(3.1169408125771287) q[16];
cx q[15],q[16];
ry(-2.544519525798625) q[17];
ry(-0.6253390932418135) q[18];
cx q[17],q[18];
ry(-2.336902909938004) q[17];
ry(3.1407616643490024) q[18];
cx q[17],q[18];
ry(2.464897862138849) q[0];
ry(-2.290855957314037) q[1];
cx q[0],q[1];
ry(-0.3456816372211353) q[0];
ry(2.35207199467456) q[1];
cx q[0],q[1];
ry(-1.1202173441161045) q[2];
ry(1.8058012963546406) q[3];
cx q[2],q[3];
ry(2.941064279460726) q[2];
ry(-2.9683756114402624) q[3];
cx q[2],q[3];
ry(2.289335778349938) q[4];
ry(2.719455504388244) q[5];
cx q[4],q[5];
ry(-0.09002337402479425) q[4];
ry(-0.030614981850697554) q[5];
cx q[4],q[5];
ry(3.1104520807521063) q[6];
ry(-3.068548865975745) q[7];
cx q[6],q[7];
ry(-1.4390080814567743) q[6];
ry(-2.399057856894232) q[7];
cx q[6],q[7];
ry(-0.49559496389260466) q[8];
ry(-0.4147450204085864) q[9];
cx q[8],q[9];
ry(-0.24743853624399395) q[8];
ry(0.088863367595728) q[9];
cx q[8],q[9];
ry(-0.7332877969687397) q[10];
ry(-1.223482343429355) q[11];
cx q[10],q[11];
ry(-3.1393118576974905) q[10];
ry(3.1372732233325555) q[11];
cx q[10],q[11];
ry(-2.319722401791623) q[12];
ry(2.2098244887896126) q[13];
cx q[12],q[13];
ry(0.031295338498682684) q[12];
ry(-3.126413368512436) q[13];
cx q[12],q[13];
ry(-2.562780691726043) q[14];
ry(2.098546679781969) q[15];
cx q[14],q[15];
ry(-2.353180602268512) q[14];
ry(0.051330844549208846) q[15];
cx q[14],q[15];
ry(-0.44653644677389887) q[16];
ry(0.661687539494552) q[17];
cx q[16],q[17];
ry(3.124129125171906) q[16];
ry(2.540229028780684) q[17];
cx q[16],q[17];
ry(-0.7613010660479853) q[18];
ry(-0.6751418215160134) q[19];
cx q[18],q[19];
ry(-2.3155947048444867) q[18];
ry(0.6396339950325797) q[19];
cx q[18],q[19];
ry(2.3616885763093265) q[1];
ry(-1.3467876343605267) q[2];
cx q[1],q[2];
ry(-0.2586847708691871) q[1];
ry(-1.520334517764672) q[2];
cx q[1],q[2];
ry(3.106621429461346) q[3];
ry(1.4394383294210376) q[4];
cx q[3],q[4];
ry(2.9832440354494465) q[3];
ry(0.030792852510779234) q[4];
cx q[3],q[4];
ry(2.606429140353495) q[5];
ry(1.2190821786690673) q[6];
cx q[5],q[6];
ry(0.0013023086738960113) q[5];
ry(-3.1225094508962297) q[6];
cx q[5],q[6];
ry(3.0240700877336497) q[7];
ry(-1.81229487401148) q[8];
cx q[7],q[8];
ry(-0.047228218658963435) q[7];
ry(0.04484880994495078) q[8];
cx q[7],q[8];
ry(-0.9021589484758612) q[9];
ry(-0.551022797626822) q[10];
cx q[9],q[10];
ry(0.12779376221730931) q[9];
ry(-2.807875854000973) q[10];
cx q[9],q[10];
ry(-2.5336698295116693) q[11];
ry(-2.271346698602843) q[12];
cx q[11],q[12];
ry(1.0325703253738636) q[11];
ry(-2.1364915047535824) q[12];
cx q[11],q[12];
ry(-1.1552904429203321) q[13];
ry(0.8680300561068703) q[14];
cx q[13],q[14];
ry(3.1160109418271267) q[13];
ry(3.1020905180757468) q[14];
cx q[13],q[14];
ry(1.2914102895867217) q[15];
ry(3.0723651441491513) q[16];
cx q[15],q[16];
ry(-3.024750818352048) q[15];
ry(-3.1224473671489505) q[16];
cx q[15],q[16];
ry(-0.5384278988435222) q[17];
ry(2.1892657735185512) q[18];
cx q[17],q[18];
ry(-0.987020757459604) q[17];
ry(-3.1320760266646586) q[18];
cx q[17],q[18];
ry(-0.5670534908664235) q[0];
ry(0.9942398686388768) q[1];
ry(-0.328509698026215) q[2];
ry(-0.6396980116636932) q[3];
ry(-3.065544425147842) q[4];
ry(-0.7780871845710449) q[5];
ry(-0.7093857437149351) q[6];
ry(-2.427586894733894) q[7];
ry(2.0880646650674803) q[8];
ry(1.1727614602047016) q[9];
ry(-1.5324942341244165) q[10];
ry(-1.5877653154490963) q[11];
ry(-2.8478191955917196) q[12];
ry(1.777181894239073) q[13];
ry(-2.5130870364413087) q[14];
ry(2.954275794543298) q[15];
ry(-2.1544672036942734) q[16];
ry(-1.4757728096953713) q[17];
ry(-0.5339313183029334) q[18];
ry(0.39320796224149035) q[19];