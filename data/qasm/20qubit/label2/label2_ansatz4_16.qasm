OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.9165114073343649) q[0];
rz(3.0673159193259485) q[0];
ry(-1.0241347722153336) q[1];
rz(-0.012390406373167906) q[1];
ry(3.141535148019065) q[2];
rz(2.539334455295279) q[2];
ry(5.7300509594071514e-05) q[3];
rz(-0.7660170119813932) q[3];
ry(-1.5763826244254375) q[4];
rz(0.1353578211910298) q[4];
ry(-1.57006157127234) q[5];
rz(3.115701115994355) q[5];
ry(-3.1381778141538073) q[6];
rz(1.0950878159322714) q[6];
ry(-3.138819365746555) q[7];
rz(0.17077187798140278) q[7];
ry(-1.5696005844465484) q[8];
rz(-1.5625837155318463) q[8];
ry(1.5269654427352268) q[9];
rz(1.792053754753323) q[9];
ry(-1.5807222302648376) q[10];
rz(-2.9941542218191946) q[10];
ry(1.5144278977438193) q[11];
rz(-0.6103637966679635) q[11];
ry(1.3419129585752945) q[12];
rz(2.2650251643263863) q[12];
ry(1.7523269589839359) q[13];
rz(0.1816626061185218) q[13];
ry(0.0016096587733533951) q[14];
rz(1.9805386238373681) q[14];
ry(-0.0016187031481908686) q[15];
rz(1.8221510935273821) q[15];
ry(-0.5266687268695397) q[16];
rz(2.161115745573328) q[16];
ry(-1.2482178270336173) q[17];
rz(1.3964406218619494) q[17];
ry(-1.0174351631145235) q[18];
rz(1.43902863993711) q[18];
ry(-1.0806410345226116) q[19];
rz(2.7023938560840435) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.502494934531398) q[0];
rz(1.278168476257275) q[0];
ry(0.8123917605261664) q[1];
rz(1.5041739023142553) q[1];
ry(-1.8368979268454675) q[2];
rz(-2.863400689103818) q[2];
ry(1.2736510420764295) q[3];
rz(-3.1268182058021443) q[3];
ry(-2.9460368744933048) q[4];
rz(3.0372493305492245) q[4];
ry(-1.379540073588567) q[5];
rz(1.8490034714248171) q[5];
ry(-2.981922082578525) q[6];
rz(-2.1588908706781647) q[6];
ry(2.200923320015413) q[7];
rz(1.8389470040338503) q[7];
ry(1.5439630461244054) q[8];
rz(-2.0588120530778875) q[8];
ry(-1.267373569319755) q[9];
rz(-2.084405507787177) q[9];
ry(0.0018348265528871364) q[10];
rz(0.4957944866471716) q[10];
ry(-0.0003706533366596432) q[11];
rz(-0.060472960838757216) q[11];
ry(-3.1386664306450838) q[12];
rz(-2.24476301082688) q[12];
ry(-2.9234944447050744) q[13];
rz(-2.1151356519595907) q[13];
ry(-3.1412028701295553) q[14];
rz(3.0154482863542835) q[14];
ry(6.98613172245288e-05) q[15];
rz(2.1802800854159834) q[15];
ry(-0.13443759404734656) q[16];
rz(-0.4226998244450801) q[16];
ry(-2.320074538834376) q[17];
rz(-1.5016342627014612) q[17];
ry(-0.12268891322320119) q[18];
rz(-2.041443155801384) q[18];
ry(-1.450342135976924) q[19];
rz(-2.081554587051693) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.36879456365863117) q[0];
rz(1.4537836584539203) q[0];
ry(1.238437051148697) q[1];
rz(-0.19647045416553954) q[1];
ry(-0.12336723287248397) q[2];
rz(-2.4525766080562947) q[2];
ry(-3.048241648336216) q[3];
rz(-0.7011963294355185) q[3];
ry(7.433895469840111e-05) q[4];
rz(-0.1436798438449214) q[4];
ry(-3.070407395266983e-05) q[5];
rz(-1.4667770680545664) q[5];
ry(3.1372186122271173) q[6];
rz(-3.098363772358701) q[6];
ry(-0.04534549502803209) q[7];
rz(-0.0851830028227785) q[7];
ry(1.5864423404313148) q[8];
rz(-2.9713366006661612) q[8];
ry(-1.5701440687896662) q[9];
rz(0.025760770038654474) q[9];
ry(-0.0015198638355498064) q[10];
rz(1.389126138234318) q[10];
ry(0.003299142252251259) q[11];
rz(-2.813762380003701) q[11];
ry(2.3184045129532462) q[12];
rz(-3.007070448990081) q[12];
ry(0.30490187405761127) q[13];
rz(-1.5014481700114617) q[13];
ry(0.0019903209837091036) q[14];
rz(3.072604488193275) q[14];
ry(0.0009032137760458525) q[15];
rz(2.5700970251099244) q[15];
ry(3.0779116500366124) q[16];
rz(-0.5362158039804803) q[16];
ry(2.300016959943053) q[17];
rz(1.4206337877132726) q[17];
ry(-1.722438443155253) q[18];
rz(2.7217810514012326) q[18];
ry(-1.7511171504891916) q[19];
rz(-0.31016061809146045) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5229144531286258) q[0];
rz(-2.943478342199622) q[0];
ry(-2.869683930997772) q[1];
rz(0.563227662557148) q[1];
ry(1.026512004935932) q[2];
rz(-1.1825196728864116) q[2];
ry(0.7832020621758264) q[3];
rz(-2.337671977719624) q[3];
ry(-3.0993684976069913) q[4];
rz(0.7627750653526265) q[4];
ry(3.1015837263958095) q[5];
rz(-0.9796532811334854) q[5];
ry(1.5642580707138072) q[6];
rz(1.5698806776097882) q[6];
ry(-3.133488200463507) q[7];
rz(-0.5364230410486749) q[7];
ry(1.9840335858810616) q[8];
rz(2.5548138425266744) q[8];
ry(2.339046064130376) q[9];
rz(-0.6854569748738086) q[9];
ry(-3.093385834624297) q[10];
rz(0.9016436260903627) q[10];
ry(0.5386887772138061) q[11];
rz(-2.1188615331585767) q[11];
ry(0.07187874910545133) q[12];
rz(-0.025798220253612676) q[12];
ry(-0.19619060181154246) q[13];
rz(-1.6089239711563457) q[13];
ry(3.1413120297864046) q[14];
rz(-1.5243349297647182) q[14];
ry(-3.141541631952728) q[15];
rz(-1.7603767380626787) q[15];
ry(2.603275213072168) q[16];
rz(-2.8787957173550724) q[16];
ry(-0.23204913847202827) q[17];
rz(0.6614032732800114) q[17];
ry(-1.4361560861060627) q[18];
rz(-1.0595483045675422) q[18];
ry(-0.3095111727929547) q[19];
rz(-0.6828455601687278) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.1254918491056927) q[0];
rz(-2.7097191856041554) q[0];
ry(-0.5838084755105748) q[1];
rz(-1.2995410604048823) q[1];
ry(0.5949507011892319) q[2];
rz(-1.9261849109729798) q[2];
ry(2.8411186567143027) q[3];
rz(3.0862062867689644) q[3];
ry(-0.28607464817796563) q[4];
rz(-0.6710389738407995) q[4];
ry(-0.43564901380535126) q[5];
rz(0.8205166342853394) q[5];
ry(-1.5594108300998748) q[6];
rz(0.6422799210949074) q[6];
ry(-0.0003268097611464071) q[7];
rz(-2.08972684495897) q[7];
ry(3.141148965065467) q[8];
rz(1.3840418006389807) q[8];
ry(0.0007184348513451511) q[9];
rz(0.9055800138574366) q[9];
ry(3.140328757969424) q[10];
rz(0.4175830340587262) q[10];
ry(-3.139288098916707) q[11];
rz(2.297276752321937) q[11];
ry(-0.03585369542130838) q[12];
rz(-3.134943787287656) q[12];
ry(3.103081574100355) q[13];
rz(2.4178751046433673) q[13];
ry(3.139544197803421) q[14];
rz(-2.611635423320014) q[14];
ry(-0.006135998361505949) q[15];
rz(1.5080838187661794) q[15];
ry(2.816475510442297) q[16];
rz(-2.16125609208118) q[16];
ry(0.669233858509199) q[17];
rz(2.517251607085104) q[17];
ry(2.410357954904301) q[18];
rz(-2.262443642422874) q[18];
ry(2.57769198550263) q[19];
rz(-2.530619377652378) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.1347099296588614) q[0];
rz(1.805418806314341) q[0];
ry(0.07067543413706812) q[1];
rz(2.1889129613759795) q[1];
ry(-0.008231649491682869) q[2];
rz(0.5546451513548617) q[2];
ry(-0.00013247971900831604) q[3];
rz(1.0249131906357747) q[3];
ry(-1.0013366745099415) q[4];
rz(0.7473281154566288) q[4];
ry(-2.1273731906726208) q[5];
rz(-1.556269885527116) q[5];
ry(3.129610964048925) q[6];
rz(1.2268336308654098) q[6];
ry(-0.43272633498608476) q[7];
rz(1.3486559959306366) q[7];
ry(0.25800680430049505) q[8];
rz(0.5836431367105704) q[8];
ry(1.3801752380578654) q[9];
rz(-0.19952669571626158) q[9];
ry(-1.5490256119474033) q[10];
rz(-0.8915786030434648) q[10];
ry(1.4173208094533214) q[11];
rz(-2.385872956042899) q[11];
ry(0.07442861032440083) q[12];
rz(-2.8765640062810487) q[12];
ry(-3.087713968255462) q[13];
rz(-0.009502562029923391) q[13];
ry(3.141318001319083) q[14];
rz(1.712825066065685) q[14];
ry(3.1415292333754774) q[15];
rz(0.8003474388235867) q[15];
ry(0.9581263109701785) q[16];
rz(1.1886353562994716) q[16];
ry(2.260405742502367) q[17];
rz(-1.3511226017468445) q[17];
ry(0.7801178727608269) q[18];
rz(-1.884298512395131) q[18];
ry(-2.755738861075594) q[19];
rz(-3.1315319992568287) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.4275017486975692) q[0];
rz(-2.5858225699016786) q[0];
ry(-1.5604952524013411) q[1];
rz(-2.6858183712147135) q[1];
ry(0.8518921368478152) q[2];
rz(1.0076810432630285) q[2];
ry(-2.429152140773661) q[3];
rz(-1.880272476619144) q[3];
ry(-0.057107379940574354) q[4];
rz(-0.578041747208558) q[4];
ry(0.09978452664308964) q[5];
rz(1.3544604718604945) q[5];
ry(2.9029186734912544) q[6];
rz(2.043961588394809) q[6];
ry(-3.1340154587352447) q[7];
rz(-2.3802554801935774) q[7];
ry(-2.305746137841095) q[8];
rz(-1.5518309728718136) q[8];
ry(-0.8379757089212845) q[9];
rz(1.555066755329591) q[9];
ry(3.0252016760540767) q[10];
rz(-1.8199296384500105) q[10];
ry(0.06472444854434212) q[11];
rz(-2.1021272912630438) q[11];
ry(-0.01782174058642738) q[12];
rz(1.7706960084567243) q[12];
ry(-1.587876826463659) q[13];
rz(2.1317052481485455) q[13];
ry(-3.112878956210157) q[14];
rz(1.7543705617538565) q[14];
ry(-1.299853649013424) q[15];
rz(-1.583512864849591) q[15];
ry(2.056333403236791) q[16];
rz(3.124657777888154) q[16];
ry(2.254507300465354) q[17];
rz(1.652487043730952) q[17];
ry(0.6936685181002165) q[18];
rz(-2.345759478005777) q[18];
ry(2.9425061285600655) q[19];
rz(-1.58390430032731) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.10182553587116949) q[0];
rz(2.9739059065244176) q[0];
ry(0.2201602009200849) q[1];
rz(1.5252210876499546) q[1];
ry(-3.1406254707660533) q[2];
rz(0.2316646156361425) q[2];
ry(-3.136966180019976) q[3];
rz(-0.8982517324017466) q[3];
ry(-0.007970681771729767) q[4];
rz(-0.9528718112761094) q[4];
ry(-3.1345131069262777) q[5];
rz(-3.0417335163344794) q[5];
ry(-0.20905847148104517) q[6];
rz(-1.111587502835154) q[6];
ry(2.1398859340758865) q[7];
rz(-2.47064302114027) q[7];
ry(-2.336845888008719) q[8];
rz(-0.8124922610651534) q[8];
ry(-2.3403765500521536) q[9];
rz(-2.42823189974369) q[9];
ry(-0.2017826784423935) q[10];
rz(3.016299634029353) q[10];
ry(0.029183291506861057) q[11];
rz(-0.3459859890164903) q[11];
ry(0.0008208732561568995) q[12];
rz(1.449555904823353) q[12];
ry(-3.1408069851852694) q[13];
rz(2.2203572180026665) q[13];
ry(-1.5239865906068824) q[14];
rz(2.9718084731031578) q[14];
ry(1.5712832361202647) q[15];
rz(-0.1273614545778655) q[15];
ry(3.1410471722131637) q[16];
rz(0.44756310862982923) q[16];
ry(0.0011825010476176345) q[17];
rz(1.9438123055093346) q[17];
ry(2.317325159188848) q[18];
rz(1.2513664184204296) q[18];
ry(2.969223363646685) q[19];
rz(-2.7947658387932783) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.00592438320406) q[0];
rz(1.1287567272730865) q[0];
ry(-2.891080816876359) q[1];
rz(-1.5602626281105691) q[1];
ry(0.9553143071494663) q[2];
rz(-1.5577556533950538) q[2];
ry(-2.2083865339133038) q[3];
rz(-3.0055583345139905) q[3];
ry(-3.060942228997364) q[4];
rz(0.1804459646088814) q[4];
ry(3.070122382601127) q[5];
rz(0.9496284962254481) q[5];
ry(-1.3334939930220304) q[6];
rz(-0.4574194217743237) q[6];
ry(-1.599589546428209) q[7];
rz(-2.888784548369601) q[7];
ry(-3.1328901322373266) q[8];
rz(-2.177171896988015) q[8];
ry(0.01069595830327863) q[9];
rz(-2.4448720565399933) q[9];
ry(-2.7918641832918016) q[10];
rz(0.9301181263525606) q[10];
ry(-2.427908780229511) q[11];
rz(-3.1228723249791503) q[11];
ry(-0.0013953970094702228) q[12];
rz(-1.5017211737088756) q[12];
ry(-0.0013184671815077715) q[13];
rz(1.6912628936779726) q[13];
ry(1.5245846267915248) q[14];
rz(2.808413641579353) q[14];
ry(-0.13528089037807892) q[15];
rz(1.6984457847021242) q[15];
ry(1.570047242511873) q[16];
rz(-0.2600073505441615) q[16];
ry(-1.5700055739495422) q[17];
rz(-1.8636811923828107) q[17];
ry(-0.10675202095783867) q[18];
rz(-2.77554304565055) q[18];
ry(1.1213677733966998) q[19];
rz(-0.4462095955401191) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.10531598002292686) q[0];
rz(-0.4558139834226129) q[0];
ry(3.0358325668646473) q[1];
rz(0.4070750614559969) q[1];
ry(3.1317297637293486) q[2];
rz(-1.819659876056403) q[2];
ry(-3.114733917075755) q[3];
rz(2.034145319870464) q[3];
ry(0.011333529212325999) q[4];
rz(0.6512205904527184) q[4];
ry(-0.005258862326570072) q[5];
rz(-1.0842506335138546) q[5];
ry(1.017119173006166) q[6];
rz(-0.38511048644870305) q[6];
ry(2.349653027444481) q[7];
rz(-0.9559859047136007) q[7];
ry(-3.1410185086814155) q[8];
rz(1.0104547380394313) q[8];
ry(0.002371923154565359) q[9];
rz(-0.9076127807182033) q[9];
ry(-1.421975789633922) q[10];
rz(0.1008046188673255) q[10];
ry(0.8088822970874058) q[11];
rz(-2.0379256033958244) q[11];
ry(0.0009064410458821115) q[12];
rz(1.203448885552452) q[12];
ry(-0.002834995965255801) q[13];
rz(0.3088360985480963) q[13];
ry(-2.720135692068843) q[14];
rz(-1.6549263085933146) q[14];
ry(-1.1526915570822993) q[15];
rz(-0.7069367574612065) q[15];
ry(1.6191812106900514) q[16];
rz(-1.8429682468889095) q[16];
ry(-1.5657945800915076) q[17];
rz(-0.8790485616993379) q[17];
ry(-0.7403685709203177) q[18];
rz(0.21449416006968658) q[18];
ry(1.1240783720737486) q[19];
rz(1.6524041503078897) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.0362712590495557) q[0];
rz(1.7945123020716043) q[0];
ry(2.5076778191579625) q[1];
rz(-2.9218778621547616) q[1];
ry(0.03908357078513446) q[2];
rz(-3.0492088773815107) q[2];
ry(0.15141614888941657) q[3];
rz(0.8636864897591604) q[3];
ry(0.8332743383630744) q[4];
rz(1.8655856434610583) q[4];
ry(1.6577901619903779) q[5];
rz(1.223170325180134) q[5];
ry(2.896608043980441) q[6];
rz(-2.6600418663736622) q[6];
ry(-0.2623794487464041) q[7];
rz(-0.8422091327032889) q[7];
ry(-0.01635856042409594) q[8];
rz(0.04109833278991282) q[8];
ry(-3.043627381868727) q[9];
rz(-1.308270216797944) q[9];
ry(-1.5543074467400069) q[10];
rz(1.6218647134459927) q[10];
ry(-3.0296060201448514) q[11];
rz(-0.5368717389861626) q[11];
ry(-2.8526129104823856) q[12];
rz(-1.9994885397048243) q[12];
ry(0.03697968720682907) q[13];
rz(2.61288874448378) q[13];
ry(-1.6915884653188993) q[14];
rz(1.4186757986713363) q[14];
ry(-3.1310173021918253) q[15];
rz(-0.5551915908879622) q[15];
ry(-0.4943203442769102) q[16];
rz(-0.04383453009781045) q[16];
ry(2.972704363157745) q[17];
rz(0.06751800786033081) q[17];
ry(-0.10516893772233389) q[18];
rz(1.4156761295803209) q[18];
ry(-3.0741412669756483) q[19];
rz(0.8016043632237526) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.8035146119052445) q[0];
rz(2.294436057909326) q[0];
ry(-0.982812413712729) q[1];
rz(-0.705997323455673) q[1];
ry(-0.0012794554859439633) q[2];
rz(-2.046266034308302) q[2];
ry(3.140857748697686) q[3];
rz(0.8682772906777787) q[3];
ry(-3.065273404075859) q[4];
rz(0.05101185558969324) q[4];
ry(-3.131627340770812) q[5];
rz(-1.6779861833024645) q[5];
ry(-3.114634086076159) q[6];
rz(-0.6043732732848146) q[6];
ry(0.01775543352596909) q[7];
rz(0.18093271085391116) q[7];
ry(-3.1334617680873174) q[8];
rz(1.2013774511137356) q[8];
ry(3.3026681132142244e-06) q[9];
rz(-0.16021356810372112) q[9];
ry(-2.6318006991028233) q[10];
rz(1.717921935076947) q[10];
ry(-2.5874696481463757) q[11];
rz(-1.6999031195859398) q[11];
ry(0.09065136639126287) q[12];
rz(-1.205245744230349) q[12];
ry(0.07888467885449414) q[13];
rz(2.3327658622408514) q[13];
ry(1.5897900199638029) q[14];
rz(-0.7530102862915361) q[14];
ry(1.5444559620756264) q[15];
rz(-0.7427758627084887) q[15];
ry(0.17667226558631643) q[16];
rz(0.9440106918132384) q[16];
ry(-0.008663069752348207) q[17];
rz(0.9165714254319841) q[17];
ry(-1.2429313453873856) q[18];
rz(1.5342353746348807) q[18];
ry(-1.859908267809149) q[19];
rz(-1.6437501968860788) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.1702255400330897) q[0];
rz(0.7420705436090547) q[0];
ry(2.0680482991657554) q[1];
rz(3.042317120747056) q[1];
ry(1.3506582000449248) q[2];
rz(1.9748229194780704) q[2];
ry(3.087214198828989) q[3];
rz(-2.4210016182762732) q[3];
ry(1.338029387581363) q[4];
rz(-2.124272160029185) q[4];
ry(3.02615020322682) q[5];
rz(3.1097302431450093) q[5];
ry(0.00665322444922456) q[6];
rz(2.78861774598223) q[6];
ry(-3.136776198659021) q[7];
rz(0.5973126901659606) q[7];
ry(-1.6122340068253083) q[8];
rz(-1.3215544125336995) q[8];
ry(-1.5798922366325037) q[9];
rz(-2.013779484649498) q[9];
ry(1.6126611793154804) q[10];
rz(-2.836792302022925) q[10];
ry(1.5731172044095123) q[11];
rz(0.3941273215229329) q[11];
ry(-2.010765439692894) q[12];
rz(-0.07595696420542275) q[12];
ry(3.0571387275273074) q[13];
rz(0.8182918163074062) q[13];
ry(-1.7054398542064328) q[14];
rz(2.526046102835933) q[14];
ry(-1.5325445913533822) q[15];
rz(-0.3054485664557056) q[15];
ry(-1.4504149723731516) q[16];
rz(-1.8792728675766408) q[16];
ry(2.262691110762512) q[17];
rz(0.9319156920209641) q[17];
ry(-3.1209053041159773) q[18];
rz(0.8200471458848747) q[18];
ry(3.1173532343139487) q[19];
rz(-0.9131253251308173) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.0489031067032015) q[0];
rz(0.5966536728037797) q[0];
ry(-3.1288731448290807) q[1];
rz(0.876003216384894) q[1];
ry(-0.5673263127106672) q[2];
rz(2.867514535670306) q[2];
ry(-0.5339854976880171) q[3];
rz(-1.6211693553070148) q[3];
ry(-0.009290621202343216) q[4];
rz(-2.284211896621368) q[4];
ry(-0.000540134112474) q[5];
rz(-2.9921942645619395) q[5];
ry(0.6881070277325998) q[6];
rz(1.8837155576420557) q[6];
ry(-1.828806987398204) q[7];
rz(-2.6550791442877273) q[7];
ry(-2.9343220466916393) q[8];
rz(-0.11188479115947794) q[8];
ry(-1.370706955569138) q[9];
rz(-1.5826652734260682) q[9];
ry(-3.0739752068606476) q[10];
rz(-0.7951627758725507) q[10];
ry(-3.058838483946054) q[11];
rz(-0.6489925593866931) q[11];
ry(-0.2927596576850231) q[12];
rz(2.4765123598180026) q[12];
ry(0.5370829617279292) q[13];
rz(1.7819484847838831) q[13];
ry(-0.7470610217891842) q[14];
rz(-0.354749349842792) q[14];
ry(1.632379206963936) q[15];
rz(-2.2767530151389064) q[15];
ry(0.7436907751914861) q[16];
rz(-0.7108046800491605) q[16];
ry(2.3737264892350765) q[17];
rz(-2.869278607261542) q[17];
ry(1.0291707428659747) q[18];
rz(-2.59897775356737) q[18];
ry(2.1103789508517634) q[19];
rz(-2.565426470965431) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-3.0695212160463936) q[0];
rz(-2.6392657562552992) q[0];
ry(-1.0239220002027345) q[1];
rz(-1.3399612716642304) q[1];
ry(2.020309939205786) q[2];
rz(-2.0618149481370374) q[2];
ry(-2.5779278320986823) q[3];
rz(0.4630366126864071) q[3];
ry(0.08456171392241567) q[4];
rz(2.4286483113132373) q[4];
ry(-3.0261704067407447) q[5];
rz(-0.8087029193161097) q[5];
ry(0.025821202905589428) q[6];
rz(-1.8315583715187946) q[6];
ry(3.1322626172990375) q[7];
rz(0.1241953828953576) q[7];
ry(-1.578632351932426) q[8];
rz(-2.12680273603878) q[8];
ry(-0.6912382394587244) q[9];
rz(0.11833650252554051) q[9];
ry(1.5685636319637863) q[10];
rz(-2.9798435385040816) q[10];
ry(-1.5730943145095564) q[11];
rz(1.1075571115134004) q[11];
ry(3.134672792779753) q[12];
rz(0.9197113045533776) q[12];
ry(-0.04894888443747102) q[13];
rz(0.8661650221033109) q[13];
ry(-0.0872030844426428) q[14];
rz(2.8094740949046746) q[14];
ry(-3.0583179008088335) q[15];
rz(-3.105563164371799) q[15];
ry(3.1066089653879647) q[16];
rz(0.6118666890371971) q[16];
ry(3.1320792759720795) q[17];
rz(-3.0664926639364976) q[17];
ry(-1.3370065386389518) q[18];
rz(-2.2847528524574012) q[18];
ry(-1.754845563605933) q[19];
rz(2.491579736421925) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.1785844764712188) q[0];
rz(3.042751945275232) q[0];
ry(2.9608754566065647) q[1];
rz(-0.909097107768163) q[1];
ry(-3.1293733646613084) q[2];
rz(0.44524703974315305) q[2];
ry(3.104192374093343) q[3];
rz(-1.0136754943764996) q[3];
ry(-3.1394942743820917) q[4];
rz(-1.9580198549853236) q[4];
ry(0.05127410052056043) q[5];
rz(-1.6385826911683905) q[5];
ry(0.0022582727066275516) q[6];
rz(-1.6510301777617202) q[6];
ry(-3.1176402862835952) q[7];
rz(-1.3768186697856883) q[7];
ry(-0.003886859230578743) q[8];
rz(-2.4476206657316197) q[8];
ry(2.719562491110193) q[9];
rz(-3.0754107257959604) q[9];
ry(3.119751077131115) q[10];
rz(-2.9370053730924703) q[10];
ry(-3.0861455474473924) q[11];
rz(1.1122929944106446) q[11];
ry(-1.7926058873640625) q[12];
rz(0.2733408322083786) q[12];
ry(-0.3704937245354669) q[13];
rz(-1.1568559859751095) q[13];
ry(1.7277588124933256) q[14];
rz(0.6208127390459461) q[14];
ry(1.2887159681691027) q[15];
rz(-0.477596632120676) q[15];
ry(-0.6253867320924181) q[16];
rz(1.5929708372306872) q[16];
ry(0.3019008026265251) q[17];
rz(-0.5528259617125526) q[17];
ry(-3.083464139567815) q[18];
rz(-0.7019921706271361) q[18];
ry(0.05158595553027601) q[19];
rz(-3.038919354568239) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.9182647979643948) q[0];
rz(2.3171643293020434) q[0];
ry(-2.369470685046361) q[1];
rz(-2.746939396707871) q[1];
ry(1.9067377891353186) q[2];
rz(-3.079516009559049) q[2];
ry(-2.3717626836398793) q[3];
rz(-2.3120390885750455) q[3];
ry(3.112506846631025) q[4];
rz(-1.4859849491055919) q[4];
ry(-3.0622759487794733) q[5];
rz(1.471501527492932) q[5];
ry(3.121308123634316) q[6];
rz(-0.22111641438628474) q[6];
ry(0.004945765420594722) q[7];
rz(1.184013284525216) q[7];
ry(1.5784434293541072) q[8];
rz(2.1431073346359995) q[8];
ry(-1.5804905072373368) q[9];
rz(-3.1313221558233892) q[9];
ry(-1.5757846102299347) q[10];
rz(-2.880632282095177) q[10];
ry(-1.579215952549107) q[11];
rz(-2.094600601043429) q[11];
ry(0.21027535607861747) q[12];
rz(-3.1113852237353643) q[12];
ry(0.9824710023271069) q[13];
rz(-2.657377354644819) q[13];
ry(-0.35475629441083356) q[14];
rz(0.614585363142996) q[14];
ry(-3.0670857332132258) q[15];
rz(1.0203500699105275) q[15];
ry(1.941436681342828) q[16];
rz(0.9098746469648232) q[16];
ry(1.9819816552565763) q[17];
rz(-0.5122900409704094) q[17];
ry(1.0229448297586656) q[18];
rz(-2.555281885242654) q[18];
ry(1.9310485556451535) q[19];
rz(-2.889854194801624) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.03425467623832734) q[0];
rz(1.0980659019356622) q[0];
ry(-3.1331087843472147) q[1];
rz(0.2644634259307402) q[1];
ry(-0.021075601480870763) q[2];
rz(1.6464998046890118) q[2];
ry(0.04235156808034635) q[3];
rz(-2.3972828929145775) q[3];
ry(0.806898089989012) q[4];
rz(2.6029989210335898) q[4];
ry(0.7905179782501) q[5];
rz(-0.9235369136544236) q[5];
ry(-1.1586958530584202) q[6];
rz(-2.4357933066482254) q[6];
ry(2.4395610706693973) q[7];
rz(1.657145379373551) q[7];
ry(1.5737093807936553) q[8];
rz(-2.5872125855717947) q[8];
ry(0.9984279138766637) q[9];
rz(-0.5757982566242666) q[9];
ry(3.131399688014376) q[10];
rz(-2.9477080188087696) q[10];
ry(-3.132226795827669) q[11];
rz(1.106161727353427) q[11];
ry(-3.1296713760387243) q[12];
rz(1.7668693730105103) q[12];
ry(-0.003097372949291355) q[13];
rz(1.031678262596484) q[13];
ry(0.04907129519477582) q[14];
rz(-0.6177781772583691) q[14];
ry(0.006667509861138399) q[15];
rz(1.4872164438884912) q[15];
ry(-3.136496143961062) q[16];
rz(-1.7802117279826717) q[16];
ry(3.1362936020157592) q[17];
rz(2.9225443444411665) q[17];
ry(-3.0942487787096473) q[18];
rz(-1.3334021027891358) q[18];
ry(-0.33763246015444426) q[19];
rz(-1.5764173341202832) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.8150979454962055) q[0];
rz(-2.998332429460848) q[0];
ry(1.0725359543976316) q[1];
rz(2.5959406356229744) q[1];
ry(-1.5486186858690907) q[2];
rz(-0.7326497638174345) q[2];
ry(1.5587669626709797) q[3];
rz(2.358625459024854) q[3];
ry(-3.141160791099526) q[4];
rz(-2.2739747191521014) q[4];
ry(-3.1359844280645994) q[5];
rz(0.6547459433921228) q[5];
ry(3.135447850116571) q[6];
rz(2.041424889947378) q[6];
ry(-3.138326611223735) q[7];
rz(2.617674211327826) q[7];
ry(0.01975174939013158) q[8];
rz(0.8934767041703832) q[8];
ry(0.02248825885521507) q[9];
rz(-2.7870948134640923) q[9];
ry(-1.578536177587214) q[10];
rz(1.4263096954920869) q[10];
ry(1.5774091927760905) q[11];
rz(1.704483666944232) q[11];
ry(-1.5457348950185246) q[12];
rz(1.6177321672529104) q[12];
ry(1.557255300024721) q[13];
rz(0.34863906081703) q[13];
ry(-0.3703534937090152) q[14];
rz(2.3868892147538867) q[14];
ry(3.096854762086791) q[15];
rz(1.8095962906731229) q[15];
ry(-0.03921422962403831) q[16];
rz(2.6474308688124584) q[16];
ry(0.0063054371735568856) q[17];
rz(3.076576527743166) q[17];
ry(0.5939232755581342) q[18];
rz(-0.7758121729136986) q[18];
ry(-1.615636145543907) q[19];
rz(1.119162700837471) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.9913645821041238) q[0];
rz(-0.04214076741436735) q[0];
ry(0.06643146250283909) q[1];
rz(-1.4910764469541984) q[1];
ry(1.4421492559292144) q[2];
rz(0.5761527894243059) q[2];
ry(1.4213029250433087) q[3];
rz(0.5424644457469565) q[3];
ry(-0.10406797763699041) q[4];
rz(0.7509142139189511) q[4];
ry(-3.038028321338397) q[5];
rz(-2.485839388548867) q[5];
ry(2.8034082199525905) q[6];
rz(-0.1762041611863143) q[6];
ry(-2.3938601369115244) q[7];
rz(2.35008339973789) q[7];
ry(-0.26012189384682544) q[8];
rz(1.0829457512954246) q[8];
ry(0.2574108166215341) q[9];
rz(-1.952555123541364) q[9];
ry(2.8054572714463637) q[10];
rz(-2.2899394836854587) q[10];
ry(0.32824231030740947) q[11];
rz(0.8765411732407431) q[11];
ry(0.7150913417953443) q[12];
rz(1.190123531976143) q[12];
ry(-0.813770400992893) q[13];
rz(1.5539977681312325) q[13];
ry(2.7868072897144076) q[14];
rz(-1.5103281662958716) q[14];
ry(-0.36883187956689834) q[15];
rz(-1.4858589809971725) q[15];
ry(-0.2957342458835601) q[16];
rz(2.570789673183467) q[16];
ry(-0.24337278831263465) q[17];
rz(2.294227460671633) q[17];
ry(-1.9990318629335881) q[18];
rz(2.3342506336283457) q[18];
ry(-1.31322206099669) q[19];
rz(-0.9578643735114885) q[19];