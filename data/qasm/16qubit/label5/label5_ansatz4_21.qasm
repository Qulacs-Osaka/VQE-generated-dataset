OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.4886280255389017) q[0];
rz(-1.7933903153812025) q[0];
ry(0.5381339321379854) q[1];
rz(-0.36683506226615586) q[1];
ry(0.8916026913307427) q[2];
rz(-1.9159857878589992) q[2];
ry(2.19338518092532) q[3];
rz(1.3709682738046185) q[3];
ry(-0.011961578422177986) q[4];
rz(1.860633584664888) q[4];
ry(0.011575104902298072) q[5];
rz(0.5925866211399685) q[5];
ry(-0.1853393868647908) q[6];
rz(-1.4281059905484745) q[6];
ry(1.2453312770639302) q[7];
rz(-2.2701473205691407) q[7];
ry(-0.0003704851076131703) q[8];
rz(-2.8065336199826434) q[8];
ry(-0.0003519741693995826) q[9];
rz(-2.7170430902496188) q[9];
ry(-0.7645181471521305) q[10];
rz(-1.8363403240218048) q[10];
ry(0.05781104906791068) q[11];
rz(0.1100040321402026) q[11];
ry(0.7710460601238864) q[12];
rz(1.6809359093340097) q[12];
ry(2.8013464654799356) q[13];
rz(-0.3408515059501163) q[13];
ry(2.8692163678087135) q[14];
rz(0.7977033208291681) q[14];
ry(-0.7018841901778512) q[15];
rz(0.18816978433615275) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.0252252188918374) q[0];
rz(-2.018347761995016) q[0];
ry(-2.628307165487994) q[1];
rz(1.7547778922304782) q[1];
ry(-2.873805412295572) q[2];
rz(2.06776740878223) q[2];
ry(-0.4118577737097353) q[3];
rz(-2.035109107318062) q[3];
ry(-0.3277726302349458) q[4];
rz(-0.18057661986094442) q[4];
ry(-0.40955048935135613) q[5];
rz(-2.0413366730610356) q[5];
ry(1.5892100697394236) q[6];
rz(-0.11929789172421192) q[6];
ry(-0.4988068814227394) q[7];
rz(-0.9266535213598459) q[7];
ry(-3.141472208570873) q[8];
rz(-0.3723131230652935) q[8];
ry(0.0008990539680890919) q[9];
rz(2.83014129064713) q[9];
ry(-1.3598211075917863) q[10];
rz(0.9143452949421977) q[10];
ry(-1.5045170595024855) q[11];
rz(-0.11132030212578936) q[11];
ry(-2.75521332330018) q[12];
rz(1.2896501344427083) q[12];
ry(-2.6798920981386565) q[13];
rz(0.060029854371593146) q[13];
ry(2.187817216376465) q[14];
rz(1.2443803376929132) q[14];
ry(0.6133206072707332) q[15];
rz(1.725822302167363) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.9743337244813732) q[0];
rz(-1.7564786854000225) q[0];
ry(2.3376468440296074) q[1];
rz(2.6512011318976345) q[1];
ry(0.30734083890909414) q[2];
rz(1.097636521408397) q[2];
ry(-2.227876273758311) q[3];
rz(-2.1662820407785874) q[3];
ry(-1.8994405661385276) q[4];
rz(0.17223796304863787) q[4];
ry(-0.7110327211540486) q[5];
rz(-1.4170401025015538) q[5];
ry(-0.7106836481979518) q[6];
rz(0.3967840014612527) q[6];
ry(2.9689472085883253) q[7];
rz(-2.5366133323623816) q[7];
ry(-1.5272522527174026) q[8];
rz(0.042740868205503275) q[8];
ry(-1.6186612041364539) q[9];
rz(0.08638604319029201) q[9];
ry(1.3564751136598254) q[10];
rz(1.9868981720587024) q[10];
ry(1.8439143612445503) q[11];
rz(0.1852041097089261) q[11];
ry(-3.121846762154384) q[12];
rz(-1.745243897842633) q[12];
ry(3.0190977954710894) q[13];
rz(-1.996910958962057) q[13];
ry(0.5256325520575753) q[14];
rz(3.008252292306693) q[14];
ry(-0.3955707857630213) q[15];
rz(0.9009351372342932) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.9369759858846735) q[0];
rz(-1.7584733103331598) q[0];
ry(2.0624869150882397) q[1];
rz(-2.599310664883813) q[1];
ry(-0.1792294634846907) q[2];
rz(0.8867925139106103) q[2];
ry(2.351354674743205) q[3];
rz(-0.1747026213874906) q[3];
ry(0.1019954221852057) q[4];
rz(-1.05796803964756) q[4];
ry(-3.1105299376460094) q[5];
rz(-3.0661128236249744) q[5];
ry(-3.1384809979627675) q[6];
rz(-2.410661456030958) q[6];
ry(-0.011049008614644194) q[7];
rz(2.458328156599896) q[7];
ry(0.006942847061095492) q[8];
rz(0.3802879467264484) q[8];
ry(0.007982517124831348) q[9];
rz(1.487577831982289) q[9];
ry(3.1156625083847334) q[10];
rz(0.5517491610776029) q[10];
ry(2.9850748472018704) q[11];
rz(-3.139611358328194) q[11];
ry(0.2202315147246301) q[12];
rz(0.4643844845942355) q[12];
ry(1.947366371553632) q[13];
rz(1.3738194990242825) q[13];
ry(-0.41965936833989714) q[14];
rz(0.23688235803409394) q[14];
ry(2.8921721162537732) q[15];
rz(-1.3134812472008752) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.0751575787203875) q[0];
rz(-0.23189987000008475) q[0];
ry(0.6272109560958381) q[1];
rz(1.45216886629842) q[1];
ry(1.931852081207941) q[2];
rz(-0.13420510989506695) q[2];
ry(-0.030721212380428575) q[3];
rz(2.485149460120155) q[3];
ry(-0.7759000423759757) q[4];
rz(0.4620169648439831) q[4];
ry(-1.8134917066572314) q[5];
rz(3.1018114780181576) q[5];
ry(-2.981481136194856) q[6];
rz(-2.747192640308945) q[6];
ry(-1.8659431193897422) q[7];
rz(-3.1089826795080104) q[7];
ry(0.050995004742044096) q[8];
rz(2.692853847986693) q[8];
ry(-1.6326267389388347) q[9];
rz(1.584076247796664) q[9];
ry(1.5490863374216897) q[10];
rz(-0.7795545753707441) q[10];
ry(1.5073989017534668) q[11];
rz(0.9294817962144685) q[11];
ry(-0.9486025615198742) q[12];
rz(2.9338631684045136) q[12];
ry(-1.3421742239050234) q[13];
rz(2.928151332608802) q[13];
ry(-0.09095840311840142) q[14];
rz(1.8091361939252373) q[14];
ry(-2.807230122973813) q[15];
rz(0.14509100009018194) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.6065084413247348) q[0];
rz(0.9586679675405635) q[0];
ry(-0.22035455262817205) q[1];
rz(-1.0098498524174548) q[1];
ry(-0.2698570563228094) q[2];
rz(0.5719823679393077) q[2];
ry(1.7353321406952897) q[3];
rz(-0.2065194331182904) q[3];
ry(2.5473572308711336) q[4];
rz(-2.6841494858182866) q[4];
ry(1.1465104340974692) q[5];
rz(0.3318662772958776) q[5];
ry(1.57591467080904) q[6];
rz(-1.7951172001883204) q[6];
ry(-1.5710642712184004) q[7];
rz(-2.2150816046459783) q[7];
ry(1.5999039622495494) q[8];
rz(-0.03818885479982281) q[8];
ry(-1.445107735320871) q[9];
rz(2.38007924681632) q[9];
ry(-0.005820074401124309) q[10];
rz(0.7892880078146196) q[10];
ry(-0.006007080607254522) q[11];
rz(0.6902203484594617) q[11];
ry(2.9640528155581483) q[12];
rz(2.301557598230574) q[12];
ry(-0.8913222199074131) q[13];
rz(-2.583234042347647) q[13];
ry(1.270359489678473) q[14];
rz(-1.4590039314359484) q[14];
ry(-0.028486465764263613) q[15];
rz(2.141184109971749) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.770215880526481) q[0];
rz(-2.0702796190712247) q[0];
ry(2.22973009181464) q[1];
rz(-1.8643790565323406) q[1];
ry(0.8104986399712327) q[2];
rz(1.0969307826616328) q[2];
ry(-2.9727595910622635) q[3];
rz(-0.015415365798914196) q[3];
ry(-1.7075455101957728) q[4];
rz(1.318551437871576) q[4];
ry(-2.690570901632153) q[5];
rz(-1.398349289427542) q[5];
ry(-3.136021716955221) q[6];
rz(1.4304725737747406) q[6];
ry(-3.1403922246359275) q[7];
rz(-2.2990274889069036) q[7];
ry(3.1366087894750105) q[8];
rz(3.1016846406631653) q[8];
ry(3.1377265982201537) q[9];
rz(2.3936803997622915) q[9];
ry(3.098266783595009) q[10];
rz(2.5031356466263794) q[10];
ry(-3.1130987218799704) q[11];
rz(-0.6692050443619675) q[11];
ry(2.347916444299248) q[12];
rz(1.938665734465567) q[12];
ry(2.385123788561895) q[13];
rz(0.7917843549089612) q[13];
ry(0.6470919369030954) q[14];
rz(-2.6054006424578806) q[14];
ry(-2.0102875920706174) q[15];
rz(1.07736923664977) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.6968953384602132) q[0];
rz(2.160823613168721) q[0];
ry(-0.324153580368555) q[1];
rz(-2.495444417447395) q[1];
ry(1.6974787037354613) q[2];
rz(2.444311813910386) q[2];
ry(-0.8721855429124227) q[3];
rz(-1.1790139012802219) q[3];
ry(1.2502871165746736) q[4];
rz(0.2383169216849419) q[4];
ry(-0.9350595719600255) q[5];
rz(0.402116536071043) q[5];
ry(1.5733367851187574) q[6];
rz(-2.7776858650518284) q[6];
ry(1.5688625496391755) q[7];
rz(0.3405328061978057) q[7];
ry(-1.2809725905607428) q[8];
rz(-1.6507231215432903) q[8];
ry(-2.0697882302575143) q[9];
rz(-1.7844823519056154) q[9];
ry(-1.5720621840158406) q[10];
rz(-0.0031579759211837195) q[10];
ry(1.568472336818) q[11];
rz(-0.01790588415004013) q[11];
ry(-0.38308184696432174) q[12];
rz(2.9210492744771446) q[12];
ry(1.2076128623284932) q[13];
rz(2.1203850587010473) q[13];
ry(3.058132489877559) q[14];
rz(2.0739788420346423) q[14];
ry(-2.292784909740702) q[15];
rz(2.013210278788983) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.495121818559962) q[0];
rz(-0.6816690063138077) q[0];
ry(2.278487018735756) q[1];
rz(-0.7310158077490068) q[1];
ry(-1.4171558801628426) q[2];
rz(0.9867180414895679) q[2];
ry(1.7894697211803776) q[3];
rz(0.21393668249757977) q[3];
ry(0.6480601556779995) q[4];
rz(2.6914043067730473) q[4];
ry(-1.5861044940987035) q[5];
rz(0.5525906101358091) q[5];
ry(0.01279035479883826) q[6];
rz(2.4808702301954138) q[6];
ry(0.0150327092550217) q[7];
rz(0.7965163773093077) q[7];
ry(-0.027305471265330006) q[8];
rz(2.563790623337835) q[8];
ry(3.1226829261447584) q[9];
rz(1.748421508173446) q[9];
ry(-1.5707569791626934) q[10];
rz(-1.5275714285030455) q[10];
ry(1.5826201385691734) q[11];
rz(1.583846880732521) q[11];
ry(-0.033606422922938514) q[12];
rz(0.8866639787733996) q[12];
ry(2.0347306580111737) q[13];
rz(1.4270936030631105) q[13];
ry(-2.1918480741970994) q[14];
rz(-1.4065153366888923) q[14];
ry(-0.061108529771184195) q[15];
rz(-0.7756781771819586) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.47886947001776914) q[0];
rz(2.2646208146373765) q[0];
ry(-0.6307618115678846) q[1];
rz(1.0864722528780122) q[1];
ry(1.6053989022128605) q[2];
rz(-1.8415688967649828) q[2];
ry(2.3692452125264376) q[3];
rz(-1.119667709191309) q[3];
ry(-0.6915003994870625) q[4];
rz(-2.773189484684785) q[4];
ry(-0.8515110451973857) q[5];
rz(0.8422183371345997) q[5];
ry(0.002870884343609603) q[6];
rz(-1.2844073322014546) q[6];
ry(3.1405674336861935) q[7];
rz(-0.42311193122860846) q[7];
ry(-0.040842722494272905) q[8];
rz(0.9614642569656412) q[8];
ry(2.9800216639675408) q[9];
rz(-0.49574041016140546) q[9];
ry(2.4237661309601246) q[10];
rz(-3.0994992830978862) q[10];
ry(1.7938598366732998) q[11];
rz(-3.1154067922161817) q[11];
ry(-0.16617822207201183) q[12];
rz(2.690343378164204) q[12];
ry(-0.3909679233811901) q[13];
rz(2.6923615542776638) q[13];
ry(-2.5108043309447994) q[14];
rz(0.9607733919613146) q[14];
ry(1.2209758490215408) q[15];
rz(1.7181615054401522) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.3788719360603316) q[0];
rz(-2.5762272876952763) q[0];
ry(0.578832097352147) q[1];
rz(3.012844593222612) q[1];
ry(1.7780745868993773) q[2];
rz(-1.9130311301555922) q[2];
ry(2.0897884316641306) q[3];
rz(2.709404147226819) q[3];
ry(-0.14817505491268188) q[4];
rz(1.1832006912809825) q[4];
ry(1.518234932077133) q[5];
rz(1.1909944068846707) q[5];
ry(1.5624748902302157) q[6];
rz(-1.1005558914677975) q[6];
ry(1.5806984280313254) q[7];
rz(3.0726480447496862) q[7];
ry(2.2531223595644816) q[8];
rz(-0.30001554356370314) q[8];
ry(-0.38885143030390645) q[9];
rz(0.6901855363576687) q[9];
ry(1.5702812580414258) q[10];
rz(2.441029879260749) q[10];
ry(-1.5793092690435395) q[11];
rz(-2.296916828878693) q[11];
ry(-1.4669978818217384) q[12];
rz(-1.9732732802320214) q[12];
ry(-0.8506670108186549) q[13];
rz(-0.7386283767642912) q[13];
ry(0.1930874928735673) q[14];
rz(1.8668528295271802) q[14];
ry(2.5746888193489177) q[15];
rz(0.45247106405428866) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.7961196451109164) q[0];
rz(-0.32140598465187953) q[0];
ry(2.330506126599892) q[1];
rz(-2.5688762171060704) q[1];
ry(-0.07467855958765872) q[2];
rz(1.14368968973068) q[2];
ry(2.852703218404327) q[3];
rz(0.9904926447366503) q[3];
ry(2.118756399953302) q[4];
rz(0.6547864967497103) q[4];
ry(-2.5177788768411875) q[5];
rz(-0.2615836624227077) q[5];
ry(-0.000729649161462446) q[6];
rz(1.8006272184145242) q[6];
ry(-0.0015710844120313538) q[7];
rz(0.739776102880529) q[7];
ry(-3.1399018998826) q[8];
rz(2.40056487930188) q[8];
ry(-3.1267294874463976) q[9];
rz(-1.62582183241707) q[9];
ry(-0.0025377318831539364) q[10];
rz(-2.426939539530865) q[10];
ry(0.003591449319173101) q[11];
rz(-0.8586430834163484) q[11];
ry(1.9591355105758232) q[12];
rz(-1.2461029419241996) q[12];
ry(2.2302026985510803) q[13];
rz(0.08207446393835127) q[13];
ry(-1.311661440775451) q[14];
rz(2.7410508652229515) q[14];
ry(0.29703426293862994) q[15];
rz(-2.7993415676717075) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.24954402815803878) q[0];
rz(1.6144258039987474) q[0];
ry(-1.7306051070919344) q[1];
rz(-2.533378489561115) q[1];
ry(-2.561713652647779) q[2];
rz(-1.043135090415257) q[2];
ry(-0.6456807456686592) q[3];
rz(-2.1862203638502997) q[3];
ry(0.533990836994545) q[4];
rz(2.342916596314366) q[4];
ry(1.864566274506747) q[5];
rz(0.892343008562797) q[5];
ry(-1.5750514906977697) q[6];
rz(0.29818625997693005) q[6];
ry(-1.575651079499357) q[7];
rz(1.8765591135281245) q[7];
ry(2.2340008631222865) q[8];
rz(2.796336326574655) q[8];
ry(1.5911540733588403) q[9];
rz(-2.8796054085192613) q[9];
ry(-1.5738803482128043) q[10];
rz(2.263271419759578) q[10];
ry(1.5746160751121234) q[11];
rz(0.11284971157557154) q[11];
ry(-0.8842379858834786) q[12];
rz(-0.8429309070465697) q[12];
ry(1.0382038962511537) q[13];
rz(0.4012465167374942) q[13];
ry(-0.4456448054350437) q[14];
rz(1.4935725701109277) q[14];
ry(-1.7258749604919368) q[15];
rz(-0.7794299094371732) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.7400783982027388) q[0];
rz(-2.6112263289380775) q[0];
ry(-0.6534567837690105) q[1];
rz(-0.8559319719485774) q[1];
ry(2.4260749923024725) q[2];
rz(0.605942942406617) q[2];
ry(2.967675560443182) q[3];
rz(2.7668292455383066) q[3];
ry(-0.3089478054088275) q[4];
rz(-0.6356034810378359) q[4];
ry(-1.6168693098900109) q[5];
rz(-2.4062588380463126) q[5];
ry(0.23192600337211988) q[6];
rz(-1.8549695822636703) q[6];
ry(-0.6472904032970859) q[7];
rz(2.7692234968729457) q[7];
ry(1.5799519155338155) q[8];
rz(1.0951019426364423) q[8];
ry(1.5542576542595614) q[9];
rz(0.38465047211434883) q[9];
ry(-0.029919006080167065) q[10];
rz(-0.6952889762064098) q[10];
ry(0.0016554513912003799) q[11];
rz(-1.7005072122256204) q[11];
ry(-0.7389399804416924) q[12];
rz(-0.5083827345103299) q[12];
ry(1.2523293162686358) q[13];
rz(0.046130327806448136) q[13];
ry(-0.32075456062987884) q[14];
rz(2.885251473902563) q[14];
ry(1.883323116735376) q[15];
rz(3.109632652726563) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.0761921350813575) q[0];
rz(-0.8337923267635096) q[0];
ry(-1.3936788046898654) q[1];
rz(1.1810267392652563) q[1];
ry(-0.4600393809180075) q[2];
rz(0.2617780153598637) q[2];
ry(-0.6665148751268362) q[3];
rz(2.5281501560867765) q[3];
ry(-1.6037815588434858) q[4];
rz(0.19912895996564342) q[4];
ry(-1.5510580224651598) q[5];
rz(0.0530314439522126) q[5];
ry(-0.9298116709953078) q[6];
rz(-1.25521927243189) q[6];
ry(1.6140660691681399) q[7];
rz(-0.7193249204635455) q[7];
ry(-0.043957249346602355) q[8];
rz(-0.4930136672106089) q[8];
ry(-1.40986892426058) q[9];
rz(1.6760970768540377) q[9];
ry(1.1047619015432035) q[10];
rz(-1.51819726034378) q[10];
ry(0.8459157900318913) q[11];
rz(1.6009459304291658) q[11];
ry(-0.7632752037764545) q[12];
rz(0.5789204685408054) q[12];
ry(2.503207053263677) q[13];
rz(-2.2997434573518003) q[13];
ry(1.256326008934279) q[14];
rz(2.100143986890197) q[14];
ry(-0.5171531776949022) q[15];
rz(0.7051095363615886) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.789811353715418) q[0];
rz(0.09806355350818396) q[0];
ry(-1.595167987865607) q[1];
rz(0.41105988527306714) q[1];
ry(-2.9470923720000197) q[2];
rz(2.955006479246722) q[2];
ry(-0.11438857625259455) q[3];
rz(-0.5526724482596309) q[3];
ry(-3.1318019671871706) q[4];
rz(-1.5206524048969101) q[4];
ry(-0.010997334034980709) q[5];
rz(2.4236443388642868) q[5];
ry(3.1413351131351632) q[6];
rz(0.32178645774382186) q[6];
ry(-3.1413850129467047) q[7];
rz(0.8465997320053215) q[7];
ry(1.5922510742655505) q[8];
rz(1.7574527182749164) q[8];
ry(-0.025281519562762256) q[9];
rz(-0.6799043248796667) q[9];
ry(1.5711116120815118) q[10];
rz(-1.5696468416545928) q[10];
ry(-1.5690047032423526) q[11];
rz(-1.570227891343203) q[11];
ry(-2.263993104282534) q[12];
rz(-1.2816074731745832) q[12];
ry(0.7507021896395214) q[13];
rz(-2.115124974833959) q[13];
ry(-1.4206114402331858) q[14];
rz(-1.1064442589085297) q[14];
ry(2.3881875197392497) q[15];
rz(-2.2401005241669174) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.4014934673785504) q[0];
rz(-2.2256820190529476) q[0];
ry(-0.06986083033930601) q[1];
rz(-0.1400280583493037) q[1];
ry(-1.0533934735886614) q[2];
rz(-1.8210503283448218) q[2];
ry(1.3913642967879971) q[3];
rz(2.62542263567148) q[3];
ry(-1.4504305614865942) q[4];
rz(-0.015152179220530469) q[4];
ry(3.104311579740976) q[5];
rz(-1.3169637710132855) q[5];
ry(1.5734199018464228) q[6];
rz(-2.209373868293956) q[6];
ry(-1.5723209273982324) q[7];
rz(2.2252841319181265) q[7];
ry(-7.404517056951545e-05) q[8];
rz(-1.764216907800824) q[8];
ry(3.1409735896193034) q[9];
rz(-0.9248509739811563) q[9];
ry(-0.30450241625510355) q[10];
rz(-3.112747593657131) q[10];
ry(2.671507260883585) q[11];
rz(0.06198978701176161) q[11];
ry(1.6095999976754696) q[12];
rz(1.680272675780305) q[12];
ry(2.5125086033881923) q[13];
rz(0.7152300940338039) q[13];
ry(2.791074684572937) q[14];
rz(-2.547919505934779) q[14];
ry(-1.1858611821024034) q[15];
rz(0.42981748234605727) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.0069916418994564) q[0];
rz(2.4485448404857055) q[0];
ry(1.1055188934571791) q[1];
rz(-0.6393058961517923) q[1];
ry(-3.1198406643898893) q[2];
rz(0.7078519939565993) q[2];
ry(3.0520407976107005) q[3];
rz(1.7797045662818105) q[3];
ry(1.8111190310603675) q[4];
rz(-1.5576799394854306) q[4];
ry(1.9628396603368037) q[5];
rz(2.0340539477472817) q[5];
ry(3.141205719787141) q[6];
rz(-2.178683449352893) q[6];
ry(3.141187620962212) q[7];
rz(-2.4892129173752133) q[7];
ry(-1.5531221922064784) q[8];
rz(-0.37551367470210906) q[8];
ry(0.02121362366401769) q[9];
rz(-2.7738879702951666) q[9];
ry(-0.06523548004671406) q[10];
rz(2.1097212778887053) q[10];
ry(0.03428858583902733) q[11];
rz(-1.112883813549927) q[11];
ry(0.01302648137846685) q[12];
rz(-1.6209782064838647) q[12];
ry(0.06680892138048203) q[13];
rz(-2.683882935498427) q[13];
ry(-3.072253915370217) q[14];
rz(0.4723101302364254) q[14];
ry(0.7671833267016801) q[15];
rz(2.287639939636313) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.9381190193600979) q[0];
rz(1.9446078358059669) q[0];
ry(0.8851553406751576) q[1];
rz(1.6484738779620631) q[1];
ry(-1.5717078065052386) q[2];
rz(-1.0059714855572681) q[2];
ry(3.126062366633007) q[3];
rz(-3.0553592149192834) q[3];
ry(-1.4904641063546251) q[4];
rz(-1.669678931533311) q[4];
ry(1.7004191590193278) q[5];
rz(-0.8699433215529178) q[5];
ry(1.5713406495165279) q[6];
rz(-1.5700341128603652) q[6];
ry(1.5701796593375967) q[7];
rz(1.4943757963035207) q[7];
ry(0.029033420025014977) q[8];
rz(-1.1222744295329088) q[8];
ry(-2.877057358936759) q[9];
rz(0.2877454656689107) q[9];
ry(0.0007780468912224548) q[10];
rz(2.216518931166415) q[10];
ry(-3.140717786448744) q[11];
rz(2.0486095195800216) q[11];
ry(2.6623540712469427) q[12];
rz(0.2510240913913453) q[12];
ry(-0.6289650221894916) q[13];
rz(-3.110800597824322) q[13];
ry(2.9483186068142775) q[14];
rz(-0.3311699890166268) q[14];
ry(-1.6411743750929118) q[15];
rz(-1.9796210314399834) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.6099034828843646) q[0];
rz(0.48881627951554407) q[0];
ry(-1.5705996727009488) q[1];
rz(-0.40048934273441555) q[1];
ry(0.011847101760439571) q[2];
rz(-0.610276440603977) q[2];
ry(-0.9017034179288945) q[3];
rz(3.110011140014361) q[3];
ry(0.004459572242498902) q[4];
rz(-1.964413610675402) q[4];
ry(-0.000720631190391785) q[5];
rz(-0.5595770600508387) q[5];
ry(1.5707463202886203) q[6];
rz(3.0804744585304475) q[6];
ry(0.0008924835816195299) q[7];
rz(-1.4927756782232076) q[7];
ry(-1.5090243547233702) q[8];
rz(1.4952337356972079) q[8];
ry(0.09843913116091585) q[9];
rz(-0.017777000021655907) q[9];
ry(0.042598884124803636) q[10];
rz(2.7490112076137296) q[10];
ry(-2.9935918446209664) q[11];
rz(-1.655603844802548) q[11];
ry(-0.5887992112807598) q[12];
rz(2.147052753173342) q[12];
ry(-1.585060891039222) q[13];
rz(-2.8630470719181846) q[13];
ry(-0.28603062451802863) q[14];
rz(0.12339980694237339) q[14];
ry(0.8011462194992232) q[15];
rz(-2.683491851694289) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-3.1086370103974126) q[0];
rz(1.13213843794254) q[0];
ry(-3.128134335244927) q[1];
rz(-1.9484400293333817) q[1];
ry(-1.652730749880087) q[2];
rz(1.5995039684236805) q[2];
ry(1.5699552766607838) q[3];
rz(-3.1404861273697664) q[3];
ry(-0.00021819147169388484) q[4];
rz(-0.7565175406726093) q[4];
ry(-0.0005530543819034506) q[5];
rz(-1.4555551281421972) q[5];
ry(-0.0023420065864143465) q[6];
rz(1.8573201564097157) q[6];
ry(-1.5703627203112687) q[7];
rz(1.5627670654024444) q[7];
ry(-0.00048141631253795225) q[8];
rz(-1.3167037434093345) q[8];
ry(-0.0008991125023936064) q[9];
rz(-0.21658331741628262) q[9];
ry(3.1392400142137062) q[10];
rz(2.38846632219052) q[10];
ry(3.141254435097557) q[11];
rz(-0.0437949223016485) q[11];
ry(-0.004395822594811209) q[12];
rz(-2.113094645608876) q[12];
ry(3.115182010930797) q[13];
rz(1.34849162822682) q[13];
ry(0.028142219105022903) q[14];
rz(1.8728486286748751) q[14];
ry(1.0324331900272474) q[15];
rz(1.5318269213498785) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.027217302180613454) q[0];
rz(-0.5791293546047599) q[0];
ry(1.5956729772816072) q[1];
rz(2.0696301452211228) q[1];
ry(1.4680808905096936) q[2];
rz(-2.427889435013919) q[2];
ry(-1.7555988383237109) q[3];
rz(-0.015908993134634744) q[3];
ry(-3.139518194862905) q[4];
rz(-2.7464529913146043) q[4];
ry(3.1403977913986774) q[5];
rz(-1.4631444319608127) q[5];
ry(0.0009929624019914265) q[6];
rz(1.4608944864854234) q[6];
ry(1.5717790825919415) q[7];
rz(1.5687900852738448) q[7];
ry(-1.571758078911675) q[8];
rz(0.00041548100174537694) q[8];
ry(1.5706874916050362) q[9];
rz(0.08139999305085618) q[9];
ry(2.073331096376852) q[10];
rz(1.565985249393564) q[10];
ry(-1.565880375540163) q[11];
rz(-3.006183685924225) q[11];
ry(2.2438396900566273) q[12];
rz(-1.4737534800033245) q[12];
ry(-0.01623757495602722) q[13];
rz(0.48625694224753785) q[13];
ry(3.127297001644938) q[14];
rz(2.6538750935027986) q[14];
ry(0.1479502987362643) q[15];
rz(1.655188415765955) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.604721896028109) q[0];
rz(-1.8279823877136585) q[0];
ry(-0.07944914347432823) q[1];
rz(2.9270683727059477) q[1];
ry(-2.9479886401799487) q[2];
rz(0.7047850549172107) q[2];
ry(2.809638130288602) q[3];
rz(-0.01500442644567812) q[3];
ry(-2.7301315264880297) q[4];
rz(2.2609236028228974) q[4];
ry(-1.764874049737859e-05) q[5];
rz(-0.5857549161443419) q[5];
ry(3.13912691745503) q[6];
rz(-1.4506867131374914) q[6];
ry(1.578685314844913) q[7];
rz(-1.6763719718898131) q[7];
ry(-1.5740630966105398) q[8];
rz(-0.0005245568439774108) q[8];
ry(-3.1375926454165275) q[9];
rz(0.9419128790000078) q[9];
ry(1.5704852347689964) q[10];
rz(1.568653325261029) q[10];
ry(1.5717974176410243) q[11];
rz(1.5736633844199264) q[11];
ry(3.140778000156654) q[12];
rz(-2.1014387305722835) q[12];
ry(1.5763221616313177) q[13];
rz(2.5296061687153784) q[13];
ry(0.03872188747102954) q[14];
rz(-0.7074474707600026) q[14];
ry(0.7620968026688147) q[15];
rz(-1.8223695154046249) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.001348564721840948) q[0];
rz(1.6611388730587537) q[0];
ry(-0.10511080001919826) q[1];
rz(2.7539178420164467) q[1];
ry(-1.5666595481049912) q[2];
rz(1.0916565710297814) q[2];
ry(-1.5802000152914895) q[3];
rz(3.1403085259873778) q[3];
ry(3.1414600654405396) q[4];
rz(2.2604552864974945) q[4];
ry(0.00012614550572020988) q[5];
rz(-2.3264153491350865) q[5];
ry(-2.982562836439993) q[6];
rz(0.003621908497908138) q[6];
ry(-3.1410480619791263) q[7];
rz(0.5555052390646156) q[7];
ry(-1.56413280127196) q[8];
rz(-1.5703375120580587) q[8];
ry(0.0020471444014145356) q[9];
rz(1.7727641821093465) q[9];
ry(0.7288346190290679) q[10];
rz(0.7107148902868374) q[10];
ry(1.573288398542278) q[11];
rz(-1.5688360531317507) q[11];
ry(-3.1406384504618274) q[12];
rz(-2.6158529481422717) q[12];
ry(-0.010926438040749572) q[13];
rz(-0.9655961742837623) q[13];
ry(-1.567563673405032) q[14];
rz(-0.828162191139393) q[14];
ry(1.494810091030433) q[15];
rz(-1.3796212338963185) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-3.0847656210157366) q[0];
rz(1.7687149304329122) q[0];
ry(-2.696179321417404) q[1];
rz(1.1077320377934008) q[1];
ry(0.03821689330489953) q[2];
rz(0.8473545606852985) q[2];
ry(-1.571070172381074) q[3];
rz(1.3637181206403655) q[3];
ry(1.5703174397036976) q[4];
rz(1.9430030331886314) q[4];
ry(-6.641700418441587e-05) q[5];
rz(2.8637496716712283) q[5];
ry(1.5705327632020873) q[6];
rz(1.9592875075569776) q[6];
ry(-0.00021571926173091782) q[7];
rz(2.40531041474954) q[7];
ry(-1.564554452859197) q[8];
rz(-1.2053260140687785) q[8];
ry(3.1414474226217957) q[9];
rz(-0.5922553245337872) q[9];
ry(-3.1413370002808736) q[10];
rz(1.0945637198529825) q[10];
ry(-1.570496208034892) q[11];
rz(-0.07436993870898931) q[11];
ry(-3.1357657146916287) q[12];
rz(2.8980916793590765) q[12];
ry(-1.5688995822303125) q[13];
rz(1.4937950176632633) q[13];
ry(-0.006640138230919478) q[14];
rz(-1.943992455038157) q[14];
ry(3.139080156614179) q[15];
rz(0.11228033906737965) q[15];