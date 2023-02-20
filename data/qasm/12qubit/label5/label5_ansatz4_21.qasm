OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.2778237667753807) q[0];
rz(2.8787827285494094) q[0];
ry(1.7672348084381977) q[1];
rz(-1.5659478168539778) q[1];
ry(1.552413522574152) q[2];
rz(-1.6671666521148198) q[2];
ry(-1.5711569300771027) q[3];
rz(-0.08000109039691504) q[3];
ry(-0.0013221894434620293) q[4];
rz(1.4844213019316583) q[4];
ry(-0.007200341128120513) q[5];
rz(2.7975987509589713) q[5];
ry(-0.39651262953317445) q[6];
rz(2.147106554453027) q[6];
ry(1.5174559596586183) q[7];
rz(2.920014964368563) q[7];
ry(-0.3480757293528054) q[8];
rz(-0.693814502111702) q[8];
ry(1.6847293857406072) q[9];
rz(-1.0036782743339572) q[9];
ry(-0.35321250598702136) q[10];
rz(-2.824858005485584) q[10];
ry(1.7557291168255) q[11];
rz(1.9426991506363207) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(3.134511507545905) q[0];
rz(2.4562964038932313) q[0];
ry(-1.5690590800151865) q[1];
rz(-1.5043384649456542) q[1];
ry(-0.0036822671075235906) q[2];
rz(1.8044410048893518) q[2];
ry(0.06966376553467271) q[3];
rz(-2.599248280410709) q[3];
ry(-3.135512977417191) q[4];
rz(3.1132786986343897) q[4];
ry(3.1400415668465804) q[5];
rz(3.05707316722009) q[5];
ry(1.122765734246804) q[6];
rz(1.7360364587408874) q[6];
ry(0.8747183730067941) q[7];
rz(0.4160691797808446) q[7];
ry(2.3241244974297537) q[8];
rz(2.7870246688661044) q[8];
ry(0.9578840509942732) q[9];
rz(-0.9931769878684092) q[9];
ry(2.9019396766794316) q[10];
rz(-2.503406469490714) q[10];
ry(-1.7764164435323704) q[11];
rz(-2.0640166917976623) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.141102094017329) q[0];
rz(-1.9991191493878606) q[0];
ry(1.568367991546839) q[1];
rz(-1.5512266560627737) q[1];
ry(-3.135622150870262) q[2];
rz(2.486954629942086) q[2];
ry(3.1162503837585125) q[3];
rz(0.4654488453957599) q[3];
ry(1.575927973772405) q[4];
rz(1.081574775644162) q[4];
ry(1.5035273613643616) q[5];
rz(-1.540971278321812) q[5];
ry(-0.6117527004690855) q[6];
rz(0.3168497337718886) q[6];
ry(-1.1849685869783053) q[7];
rz(-0.04499661700518616) q[7];
ry(1.271785351192169) q[8];
rz(-0.8239210090249766) q[8];
ry(-1.1648929366608503) q[9];
rz(2.2824913821415107) q[9];
ry(-1.4452908602844303) q[10];
rz(0.48753360086381026) q[10];
ry(2.2785712973489494) q[11];
rz(1.8626247897670638) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.2866890781684872) q[0];
rz(2.0145660790085205) q[0];
ry(0.19091428640059344) q[1];
rz(2.9966364053492325) q[1];
ry(0.005304867533048165) q[2];
rz(0.14111110325243886) q[2];
ry(-1.692958586677627) q[3];
rz(-1.5563688293361038) q[3];
ry(-0.08966987520736319) q[4];
rz(0.04394132981288834) q[4];
ry(2.4039332178507906) q[5];
rz(3.1149731961559564) q[5];
ry(3.1413812749295262) q[6];
rz(-1.842639922622161) q[6];
ry(-3.105598882055262) q[7];
rz(0.39797361819497445) q[7];
ry(1.9678122613805784) q[8];
rz(-2.872896298893941) q[8];
ry(1.4560493064162996) q[9];
rz(-2.0592248148423495) q[9];
ry(-0.26050021225244013) q[10];
rz(2.7463333821968385) q[10];
ry(1.6966745525417666) q[11];
rz(1.8408179847799921) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.8768594394113213) q[0];
rz(-2.0188764598133337) q[0];
ry(-1.6112639916018658) q[1];
rz(0.8168026128932846) q[1];
ry(0.03961135767928514) q[2];
rz(1.1183949105376447) q[2];
ry(1.601773758071823) q[3];
rz(1.1362831386116592) q[3];
ry(0.14943565800776232) q[4];
rz(0.821850542818651) q[4];
ry(1.4194746517051833) q[5];
rz(-1.626833720641602) q[5];
ry(1.576843351194004) q[6];
rz(-1.5318204535992428) q[6];
ry(-1.5559805807904281) q[7];
rz(-1.5817074911198556) q[7];
ry(-2.2334456412715418) q[8];
rz(2.5528509852547896) q[8];
ry(-0.5250258301638614) q[9];
rz(2.065665335904251) q[9];
ry(-2.941424483748137) q[10];
rz(-0.1671003051311306) q[10];
ry(-2.321292862516681) q[11];
rz(-1.9535663809172246) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.20409331971105968) q[0];
rz(1.6570333005388695) q[0];
ry(0.06719077675422991) q[1];
rz(0.5263853874093735) q[1];
ry(-0.0457697830962811) q[2];
rz(-3.0586432255804175) q[2];
ry(3.0978463814538344) q[3];
rz(0.2623389254449826) q[3];
ry(-0.00898804623430216) q[4];
rz(1.5923883607504798) q[4];
ry(3.12666910152565) q[5];
rz(-1.8098051774697312) q[5];
ry(0.5236426679060626) q[6];
rz(3.092002107256143) q[6];
ry(2.8016945537507314) q[7];
rz(3.137745678761867) q[7];
ry(0.8738610232176596) q[8];
rz(2.9521506637514188) q[8];
ry(1.1445822444704339) q[9];
rz(0.3647317136565986) q[9];
ry(1.2436027769772375) q[10];
rz(2.3974734148518526) q[10];
ry(-3.000594640928643) q[11];
rz(1.9311974012095054) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.8579587754586573) q[0];
rz(0.24650096159574897) q[0];
ry(-2.6168472384352026) q[1];
rz(-1.5042120945085034) q[1];
ry(-3.080566087717559) q[2];
rz(0.5436087022545315) q[2];
ry(3.117569830329155) q[3];
rz(-2.465693733390716) q[3];
ry(-0.010628037229380969) q[4];
rz(2.266603156602357) q[4];
ry(-2.997463587830781) q[5];
rz(1.9058344518177581) q[5];
ry(2.665697294704085) q[6];
rz(2.965719257001407) q[6];
ry(0.6375327383501297) q[7];
rz(-0.12272329210887911) q[7];
ry(2.754964369863299) q[8];
rz(1.5086495602518006) q[8];
ry(2.437429407022548) q[9];
rz(-2.877756009017199) q[9];
ry(1.5131814280990579) q[10];
rz(1.7577897987590507) q[10];
ry(-2.9299633537374628) q[11];
rz(-1.5984858909892625) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.365132885744965) q[0];
rz(0.6121739543841666) q[0];
ry(0.2565750131403738) q[1];
rz(0.38603660876092327) q[1];
ry(-1.8242565786343958) q[2];
rz(2.587007520746687) q[2];
ry(-1.8162474852013542) q[3];
rz(2.373907707985427) q[3];
ry(1.6021363962048305) q[4];
rz(-1.5527103112690384) q[4];
ry(0.0731958810614648) q[5];
rz(1.1739895966088953) q[5];
ry(0.05239808048613942) q[6];
rz(-1.3660338933588339) q[6];
ry(2.98798650411662) q[7];
rz(1.4425198260641297) q[7];
ry(1.7737704268790324) q[8];
rz(-1.3450128962637269) q[8];
ry(-2.229888211433784) q[9];
rz(2.1290185763535807) q[9];
ry(1.6006524279006475) q[10];
rz(0.7073147907776355) q[10];
ry(-1.8236162937344664) q[11];
rz(3.0103184948418797) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.731734263356641) q[0];
rz(2.9137434782149345) q[0];
ry(-1.259609876914129) q[1];
rz(-2.6546065733594193) q[1];
ry(-1.6128997731776826) q[2];
rz(-3.087311091679463) q[2];
ry(1.520106937591444) q[3];
rz(3.0858952833636) q[3];
ry(2.489435440950376) q[4];
rz(-0.5034169047921183) q[4];
ry(-0.004865245719855693) q[5];
rz(1.4169130378766424) q[5];
ry(0.9088149966410537) q[6];
rz(-1.596585246451732) q[6];
ry(-2.8026796744865736) q[7];
rz(1.545771328948092) q[7];
ry(1.690142641393666) q[8];
rz(-0.2644512911505896) q[8];
ry(-2.064278183645573) q[9];
rz(-2.855872271395509) q[9];
ry(-2.5656929008351934) q[10];
rz(0.5067424340013013) q[10];
ry(0.6286142869034149) q[11];
rz(1.1117628918812583) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.7666024995901513) q[0];
rz(-2.346788246131095) q[0];
ry(-1.3624298729649882) q[1];
rz(-1.954651760761254) q[1];
ry(1.464239453449876) q[2];
rz(2.245720479380429) q[2];
ry(1.6667719477350784) q[3];
rz(1.998225827828908) q[3];
ry(3.1276531673471877) q[4];
rz(-0.2660063141903848) q[4];
ry(-3.12555197337529) q[5];
rz(0.9297247172893134) q[5];
ry(2.7051268371453063) q[6];
rz(1.635621292152928) q[6];
ry(-0.557253778147034) q[7];
rz(1.5582354688950808) q[7];
ry(0.9648545428684491) q[8];
rz(1.6297618667036022) q[8];
ry(-0.7571063467236065) q[9];
rz(-1.5822608849890683) q[9];
ry(-0.9015547245892561) q[10];
rz(-0.28976293085257315) q[10];
ry(2.713024869695395) q[11];
rz(1.5698643735857214) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.2550391770300973) q[0];
rz(-2.2559718166592644) q[0];
ry(-0.5685102798312247) q[1];
rz(0.13106489872300803) q[1];
ry(-1.5814692419547518) q[2];
rz(-2.3855970316294886) q[2];
ry(1.4662305248762753) q[3];
rz(-2.7413250123104613) q[3];
ry(-0.0007009362343157995) q[4];
rz(-2.8281038845118953) q[4];
ry(3.124549159318914) q[5];
rz(0.6040256540294493) q[5];
ry(0.7512986414694204) q[6];
rz(3.120499677027593) q[6];
ry(-1.285557278957557) q[7];
rz(-3.121862476122093) q[7];
ry(-2.2578205167431142) q[8];
rz(-1.1795893775911779) q[8];
ry(-1.888374923143072) q[9];
rz(-0.5342051805704467) q[9];
ry(2.495825980647493) q[10];
rz(1.3403552825308003) q[10];
ry(2.094195479669718) q[11];
rz(-2.45480721780605) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.09560084994160271) q[0];
rz(1.5375454600594143) q[0];
ry(3.1406727653953417) q[1];
rz(0.3913789352284142) q[1];
ry(-3.138220703998372) q[2];
rz(0.5246374232396684) q[2];
ry(-3.132434506894081) q[3];
rz(-1.0098837248820645) q[3];
ry(-3.130527424655719) q[4];
rz(0.3010323415822711) q[4];
ry(-3.1404162979299484) q[5];
rz(0.7940764923512922) q[5];
ry(-1.5782130890867145) q[6];
rz(2.518589206693202) q[6];
ry(-1.5897740030805814) q[7];
rz(2.2880513104856304) q[7];
ry(2.356430808987444) q[8];
rz(0.584372280133648) q[8];
ry(0.39467801610055886) q[9];
rz(1.7522685972893703) q[9];
ry(1.7269026521902768) q[10];
rz(1.746575715825057) q[10];
ry(-1.3034919759450014) q[11];
rz(-0.40866589253281166) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.6218324075178243) q[0];
rz(1.6524959878309327) q[0];
ry(-0.10885957776517864) q[1];
rz(2.6718775223945346) q[1];
ry(-1.0597925122336809) q[2];
rz(0.6844809391815749) q[2];
ry(-1.2829465341990067) q[3];
rz(-1.5794366121668462) q[3];
ry(-0.01579829731942617) q[4];
rz(2.2128190645695978) q[4];
ry(0.023702536076667346) q[5];
rz(0.4000129418273746) q[5];
ry(1.6018464750354353) q[6];
rz(1.5830709930978362) q[6];
ry(-3.1078663180361636) q[7];
rz(-2.7407977779587447) q[7];
ry(-1.8730734277696053) q[8];
rz(-1.4919011855806117) q[8];
ry(-2.611755464828702) q[9];
rz(1.4860781253531314) q[9];
ry(-1.0014402957023485) q[10];
rz(0.5685942374096841) q[10];
ry(0.9226791198980262) q[11];
rz(1.0480935891553271) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5830815466742867) q[0];
rz(0.13322237656750993) q[0];
ry(-3.140934430285394) q[1];
rz(1.6966302755129008) q[1];
ry(1.3529123325192254) q[2];
rz(1.5555281760257482) q[2];
ry(-1.6107366676645412) q[3];
rz(-0.5650806358131036) q[3];
ry(3.1388944723390146) q[4];
rz(-2.5244802802384054) q[4];
ry(0.002352192949430787) q[5];
rz(1.749144374810921) q[5];
ry(-2.648397301010486) q[6];
rz(1.5791594654903278) q[6];
ry(-1.6369347095244426) q[7];
rz(-0.9180619355065954) q[7];
ry(4.444446307447507e-05) q[8];
rz(-3.0702548801945837) q[8];
ry(-3.1197223403704917) q[9];
rz(-1.9160724977257484) q[9];
ry(-2.3017355860181716) q[10];
rz(-1.5012143031714806) q[10];
ry(2.1650513334814825) q[11];
rz(-0.7106885676357431) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.0337353202905772) q[0];
rz(-3.009661759241184) q[0];
ry(-0.013276390812393402) q[1];
rz(3.1241219638292868) q[1];
ry(0.5809590426370628) q[2];
rz(1.5872210820425892) q[2];
ry(2.1328219654587475) q[3];
rz(2.060030269736048) q[3];
ry(0.0006052955407609417) q[4];
rz(-0.2655644181421928) q[4];
ry(0.0038341345338957413) q[5];
rz(2.8356695245198744) q[5];
ry(-1.5449659781324092) q[6];
rz(0.1243759194027554) q[6];
ry(0.015872094197217912) q[7];
rz(-1.475929838804909) q[7];
ry(1.4843134889118315) q[8];
rz(0.07125864977192364) q[8];
ry(-0.33869251706872827) q[9];
rz(2.973561296429337) q[9];
ry(-2.8215403183730094) q[10];
rz(2.036768134526673) q[10];
ry(0.18889827717713922) q[11];
rz(-2.917112051851647) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5825564486944361) q[0];
rz(0.014120626972573545) q[0];
ry(-0.0008193927280508717) q[1];
rz(2.830597647669419) q[1];
ry(2.858097253656114) q[2];
rz(-2.169021378378038) q[2];
ry(-2.7891250379485513) q[3];
rz(2.1180487699095436) q[3];
ry(-1.5737353255408753) q[4];
rz(0.31054138584656416) q[4];
ry(-1.5678336536941355) q[5];
rz(-2.3270355350451415) q[5];
ry(-0.01418128667474683) q[6];
rz(-2.5734287842398524) q[6];
ry(-3.1339605796229137) q[7];
rz(-2.749622341374264) q[7];
ry(-1.0843203609048881) q[8];
rz(2.7645054474639923) q[8];
ry(-2.4381012447305355) q[9];
rz(-0.4203779697737006) q[9];
ry(-1.6794817536151463) q[10];
rz(2.008812964984349) q[10];
ry(-0.3717590843771053) q[11];
rz(-2.8468866312659276) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.1103235713591362) q[0];
rz(1.5724605899959236) q[0];
ry(1.5830285799157338) q[1];
rz(-0.12723882850498308) q[1];
ry(0.0033237543771802436) q[2];
rz(2.130093518587689) q[2];
ry(0.008857854293863987) q[3];
rz(0.2592706059379619) q[3];
ry(-3.121390443295847) q[4];
rz(-1.4573624630265787) q[4];
ry(-0.02382653283212979) q[5];
rz(0.6719181769745077) q[5];
ry(2.989594650408129) q[6];
rz(0.6421740981366446) q[6];
ry(0.15171144579128984) q[7];
rz(1.3801619636064317) q[7];
ry(0.49726043446459145) q[8];
rz(2.6247337968415807) q[8];
ry(2.3199946682203927) q[9];
rz(-2.245646877836835) q[9];
ry(2.7266065034822953) q[10];
rz(-0.9159050020828883) q[10];
ry(1.0006670289153865) q[11];
rz(0.3611757253759995) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.0030660532762469828) q[0];
rz(1.013893257322886) q[0];
ry(3.1110075020889725) q[1];
rz(1.4280937499258437) q[1];
ry(-0.2888654356796847) q[2];
rz(-0.5538271793416074) q[2];
ry(-2.9081689394632444) q[3];
rz(2.736698482580081) q[3];
ry(-3.1220278506366173) q[4];
rz(-1.7683745830943316) q[4];
ry(0.15215963065341764) q[5];
rz(-3.060692085790324) q[5];
ry(0.0032881933975508204) q[6];
rz(2.4898770330069424) q[6];
ry(-0.006355214962249298) q[7];
rz(2.2849996095348346) q[7];
ry(-0.41064421217430763) q[8];
rz(-0.8868798847069934) q[8];
ry(2.683644741939652) q[9];
rz(-0.07580354530656762) q[9];
ry(1.8044714269169757) q[10];
rz(-0.39220809675757984) q[10];
ry(1.1801586971386104) q[11];
rz(-2.7914219401742213) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.803865648763968) q[0];
rz(-1.6944035882372255) q[0];
ry(-2.850560783056136) q[1];
rz(-1.9976623811389929) q[1];
ry(-3.1414883535413796) q[2];
rz(1.107153080220864) q[2];
ry(3.1347639466245667) q[3];
rz(-2.748657966188879) q[3];
ry(1.5693163193703255) q[4];
rz(-1.5812562518845727) q[4];
ry(-1.5850360336217653) q[5];
rz(-0.020034703858833363) q[5];
ry(-0.10862572876851129) q[6];
rz(-2.469328396490842) q[6];
ry(0.11400243455693554) q[7];
rz(0.08053463513358848) q[7];
ry(-0.39894886222189546) q[8];
rz(-1.8512162833771424) q[8];
ry(-1.5858583618482283) q[9];
rz(0.9901639923063951) q[9];
ry(-1.887881406545314) q[10];
rz(0.9296582961792489) q[10];
ry(3.0950504648076977) q[11];
rz(-1.7480611245393889) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5841578008350226) q[0];
rz(-0.007387852269324035) q[0];
ry(0.0005455339150081026) q[1];
rz(-1.2817836294693858) q[1];
ry(-3.0860597514226975) q[2];
rz(3.113255864467722) q[2];
ry(-1.6025414739273456) q[3];
rz(1.757061701509099) q[3];
ry(1.5808117322655841) q[4];
rz(-2.1146319987603355) q[4];
ry(1.567502020607908) q[5];
rz(1.4849555112433324) q[5];
ry(-1.570708500647683) q[6];
rz(3.1212628128864717) q[6];
ry(-1.5591550186753018) q[7];
rz(-0.0012519505180701538) q[7];
ry(2.469077263137298) q[8];
rz(0.48504252465312164) q[8];
ry(0.785553650430873) q[9];
rz(0.8374804635739022) q[9];
ry(-2.05534958236243) q[10];
rz(-1.8525718525610708) q[10];
ry(-0.8032879801523438) q[11];
rz(0.6587845199341247) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.6194521757831624) q[0];
rz(-0.5905011578447744) q[0];
ry(-3.1215954634591583) q[1];
rz(-0.10658662658412556) q[1];
ry(-0.036877410327568985) q[2];
rz(-1.4199170115854256) q[2];
ry(-0.0024693653748081535) q[3];
rz(-1.7586133996571838) q[3];
ry(3.135063579725864) q[4];
rz(-2.1115673030295907) q[4];
ry(-0.00026587339660501193) q[5];
rz(-1.4932180321391173) q[5];
ry(-1.570965143835453) q[6];
rz(3.1414393917318626) q[6];
ry(2.0746639326321423) q[7];
rz(-0.0006583396797523733) q[7];
ry(0.6714018168618132) q[8];
rz(0.5269061331420061) q[8];
ry(-2.913562428409422) q[9];
rz(1.3346344328705588) q[9];
ry(-1.8346596369374941) q[10];
rz(3.071308743787152) q[10];
ry(0.3185895689965621) q[11];
rz(-2.4122828024526872) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.140675601397017) q[0];
rz(0.9768694974758751) q[0];
ry(-3.0675710413783075) q[1];
rz(1.5899142826115409) q[1];
ry(2.8449567842819476) q[2];
rz(-1.5503656609350687) q[2];
ry(2.595115912209208) q[3];
rz(0.00365630272559958) q[3];
ry(-1.5672757660052126) q[4];
rz(-0.6570159446142636) q[4];
ry(-1.5735986188688442) q[5];
rz(1.4060536657182858) q[5];
ry(1.5681041162958769) q[6];
rz(3.0844415407723424) q[6];
ry(1.5810965154887897) q[7];
rz(2.834219083399607) q[7];
ry(0.01778407269029647) q[8];
rz(-0.6456295789012413) q[8];
ry(-0.09212036863393269) q[9];
rz(0.0464588120721805) q[9];
ry(2.5769883122208257) q[10];
rz(-1.9947295371287224) q[10];
ry(0.8651263970879235) q[11];
rz(-0.05776686224329403) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.5815967423757604) q[0];
rz(-1.6188190453720557) q[0];
ry(-1.566652749819732) q[1];
rz(1.6236790624549213) q[1];
ry(-1.5661358890492354) q[2];
rz(1.0656098243662138) q[2];
ry(-1.5487725301803097) q[3];
rz(-1.8673040078918441) q[3];
ry(-1.6002223601363244) q[4];
rz(-2.93557693785229) q[4];
ry(-1.4123468661951506) q[5];
rz(-1.2690556941719942) q[5];
ry(-0.09445183631090551) q[6];
rz(2.7061103393327843) q[6];
ry(0.001508982712834665) q[7];
rz(3.0783542095202234) q[7];
ry(-1.552342691993356) q[8];
rz(-1.590710948933352) q[8];
ry(-3.088931414339185) q[9];
rz(1.6372809351290831) q[9];
ry(1.387524196892853) q[10];
rz(-0.5540563324654553) q[10];
ry(-1.7617968692802681) q[11];
rz(2.551364350650008) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.5322231971844067) q[0];
rz(2.084503576971143) q[0];
ry(1.5706976588301504) q[1];
rz(2.8511482992971056) q[1];
ry(3.139977408393017) q[2];
rz(-0.7714562046400529) q[2];
ry(-3.1338873431284675) q[3];
rz(-1.8716956975762047) q[3];
ry(3.140157432970781) q[4];
rz(-2.9332889176137016) q[4];
ry(-3.1397516737876985) q[5];
rz(-2.0944954922143983) q[5];
ry(3.135697189649699) q[6];
rz(-0.658295841283822) q[6];
ry(3.1281800233840897) q[7];
rz(1.1948007512312717) q[7];
ry(-0.2379996025008415) q[8];
rz(-0.6398254420673982) q[8];
ry(2.6727176941771598) q[9];
rz(2.980393524782823) q[9];
ry(-3.1404381897756175) q[10];
rz(-2.126462204501439) q[10];
ry(1.5925484597940707) q[11];
rz(-1.6022753984165334) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.06834172072949922) q[0];
rz(0.9550625301844616) q[0];
ry(-0.0011820974212221665) q[1];
rz(-0.4226809426322481) q[1];
ry(3.132764717435931) q[2];
rz(-1.9384466991422837) q[2];
ry(-1.5736319852693814) q[3];
rz(0.8474528210375399) q[3];
ry(1.538844106171781) q[4];
rz(0.5565132663439067) q[4];
ry(0.23248878100863024) q[5];
rz(-3.0444360905160455) q[5];
ry(-0.08710014356420093) q[6];
rz(-3.0929645239411485) q[6];
ry(1.5717490217535808) q[7];
rz(-2.2704992517243365) q[7];
ry(3.1209487283439716) q[8];
rz(-0.7053164472570979) q[8];
ry(1.6844946948634139) q[9];
rz(0.3632796689178406) q[9];
ry(1.5604420147166518) q[10];
rz(-0.060721423627804055) q[10];
ry(-0.020190751411399255) q[11];
rz(-2.247037748804205) q[11];