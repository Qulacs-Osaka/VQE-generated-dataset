OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.680287814895694) q[0];
rz(-0.9513066564681713) q[0];
ry(-2.5148235482095167) q[1];
rz(2.1359307685426927) q[1];
ry(-3.1415905410612415) q[2];
rz(2.3545767816088436) q[2];
ry(1.570806761576149) q[3];
rz(1.570653932101429) q[3];
ry(2.3237789784705067) q[4];
rz(0.06418512204247939) q[4];
ry(-3.1415852538713707) q[5];
rz(-0.10196077604042575) q[5];
ry(-1.1390732882020806) q[6];
rz(-3.010357487345972) q[6];
ry(3.14149551910102) q[7];
rz(1.2702417723987498) q[7];
ry(3.14154647902305) q[8];
rz(-3.127356114204153) q[8];
ry(2.3731655854806815) q[9];
rz(1.584410464481582) q[9];
ry(0.08301557354269093) q[10];
rz(-3.0864113259541166) q[10];
ry(1.1995201545162022) q[11];
rz(1.7169392206740404) q[11];
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
ry(2.879261485223601) q[0];
rz(-2.973427281066256) q[0];
ry(-3.141582596376133) q[1];
rz(-2.7458081685082396) q[1];
ry(0.7836318215682645) q[2];
rz(0.6377281444341192) q[2];
ry(-3.056839013641328) q[3];
rz(-1.5709338751127042) q[3];
ry(0.7599372624448651) q[4];
rz(0.11218883213007215) q[4];
ry(-1.6778674845843317e-06) q[5];
rz(2.5720387358949823) q[5];
ry(-1.0926731445137716) q[6];
rz(2.4898350837870646) q[6];
ry(0.00032074438137418975) q[7];
rz(0.4466724255919346) q[7];
ry(-3.1415551189282658) q[8];
rz(2.8295998075395365) q[8];
ry(2.9363276671956484) q[9];
rz(1.5944595646734683) q[9];
ry(1.3402200678410292) q[10];
rz(-1.5850715513946936) q[10];
ry(1.7127449786012605) q[11];
rz(-0.001861022518069966) q[11];
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
ry(1.5708332168502226) q[0];
rz(2.220176549847112) q[0];
ry(-1.311311616863776) q[1];
rz(1.6069184736494417) q[1];
ry(1.5707937243432024) q[2];
rz(1.5707970178781263) q[2];
ry(1.570805844011562) q[3];
rz(-0.40000809347369976) q[3];
ry(1.569585525740931) q[4];
rz(-1.4492621955608351) q[4];
ry(0.7311708864888625) q[5];
rz(-0.8267823609433625) q[5];
ry(-1.9998919549747858) q[6];
rz(3.091466852851563) q[6];
ry(-3.141503945071761) q[7];
rz(2.0255517503957354) q[7];
ry(3.1415836903943175) q[8];
rz(1.3988402694029063) q[8];
ry(-1.441841212664778) q[9];
rz(1.8186596843191516) q[9];
ry(-2.3031451980023427) q[10];
rz(0.9207337293765) q[10];
ry(-1.5570027118467245) q[11];
rz(3.0859291382057954) q[11];
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
ry(-1.1181592078912672e-06) q[0];
rz(-0.6494231024791572) q[0];
ry(1.8152860434647167) q[1];
rz(2.7828308215595134) q[1];
ry(-1.570790611122371) q[2];
rz(-1.5707933342463862) q[2];
ry(-5.939808183455412e-06) q[3];
rz(2.124104830105134) q[3];
ry(-3.141586443120284) q[4];
rz(1.5415825715360296) q[4];
ry(1.6924828367054565) q[5];
rz(1.6820680607473197) q[5];
ry(4.288076015157571e-06) q[6];
rz(-0.7982835543469236) q[6];
ry(3.1415049215115256) q[7];
rz(-0.03176728471954248) q[7];
ry(-3.1412257874900296) q[8];
rz(2.510829943364216) q[8];
ry(1.4939691954964143) q[9];
rz(0.3122119933301306) q[9];
ry(-0.36101406163134203) q[10];
rz(-1.3881616726003616) q[10];
ry(-2.18183384741213) q[11];
rz(1.3599317866522418) q[11];
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
ry(-1.789385454240559) q[0];
rz(-3.1415883805289306) q[0];
ry(2.1328782022552426) q[1];
rz(-1.6304448593625036) q[1];
ry(-1.570794341543951) q[2];
rz(-3.104897215696147) q[2];
ry(1.570805365280524) q[3];
rz(1.5707974665283748) q[3];
ry(4.2253394065773885e-06) q[4];
rz(-1.420085474581403) q[4];
ry(-1.7515751681918488) q[5];
rz(2.1268095980076724) q[5];
ry(1.9612038748531502) q[6];
rz(2.602136814082946) q[6];
ry(1.5707919383754252) q[7];
rz(-2.555371606343631) q[7];
ry(-1.0456542353397928) q[8];
rz(1.5711887673324485) q[8];
ry(-3.1345587080935426) q[9];
rz(2.723912602835256) q[9];
ry(3.025081816802259) q[10];
rz(1.1927770805497457) q[10];
ry(1.4375643730414578) q[11];
rz(1.5892786726650499) q[11];
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
ry(1.570779736277828) q[0];
rz(-2.2139476131591156) q[0];
ry(2.540993138566705e-06) q[1];
rz(0.3519568821473345) q[1];
ry(-1.7437879472836926e-05) q[2];
rz(0.2418104648719275) q[2];
ry(2.525152881268335) q[3];
rz(1.5708019679449787) q[3];
ry(1.5707936743522106) q[4];
rz(0.1616765552720203) q[4];
ry(-1.5707931330921763) q[5];
rz(2.423780007095959) q[5];
ry(-3.14158556194223) q[6];
rz(0.7260142650861249) q[6];
ry(3.141141998748011) q[7];
rz(-2.5517526169728058) q[7];
ry(1.5708030387332084) q[8];
rz(0.20993252557561706) q[8];
ry(-2.093811720534) q[9];
rz(1.5707854595252009) q[9];
ry(-1.5707978614179385) q[10];
rz(1.5704934898417813) q[10];
ry(1.6785725241469729) q[11];
rz(-2.381078006540877) q[11];
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
ry(1.7032409787952298) q[0];
rz(-1.8641901031419303) q[0];
ry(-1.5708019913960891) q[1];
rz(1.570791230832442) q[1];
ry(9.806048248961474e-07) q[2];
rz(-2.765114844626679) q[2];
ry(1.5707810354863012) q[3];
rz(-5.103753819081831e-06) q[3];
ry(-3.137222964828353) q[4];
rz(0.38515462564029423) q[4];
ry(3.1415926286418543) q[5];
rz(-0.44352438362762653) q[5];
ry(-1.5702618575958684) q[6];
rz(-0.8655827467058155) q[6];
ry(-1.5749759910554935) q[7];
rz(-1.5722263429588046) q[7];
ry(0.38449289188578856) q[8];
rz(0.8566360821593094) q[8];
ry(-1.57080642657911) q[9];
rz(0.31667562858090287) q[9];
ry(-1.570793963628358) q[10];
rz(5.463042055264822e-05) q[10];
ry(3.141556904740553) q[11];
rz(-3.0842331718438976) q[11];
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
ry(-0.9775317610687565) q[0];
rz(1.7172646461979584) q[0];
ry(1.5707948691480378) q[1];
rz(-2.193516965792526) q[1];
ry(-8.645167035445931e-07) q[2];
rz(0.6146756877052663) q[2];
ry(-1.5708013957092206) q[3];
rz(-1.4229398597435607) q[3];
ry(-0.8501281361546376) q[4];
rz(0.5289392946163378) q[4];
ry(0.0004124774710463931) q[5];
rz(-2.937414912157981) q[5];
ry(-0.0006722318160810581) q[6];
rz(-1.5616560054320536) q[6];
ry(-1.895285437179103) q[7];
rz(-1.4608217693553192) q[7];
ry(1.2812627785407926e-06) q[8];
rz(-1.0765238975025149) q[8];
ry(-4.996674327628625e-07) q[9];
rz(-1.8878847656637863) q[9];
ry(-1.060986846056883) q[10];
rz(-1.2583863818493721) q[10];
ry(1.5706926821540748) q[11];
rz(0.7672298437657971) q[11];
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
ry(-1.542527860342929) q[0];
rz(3.1415911234165232) q[0];
ry(1.6266486921878709e-06) q[1];
rz(-2.597873017720473) q[1];
ry(-0.00015300132227213933) q[2];
rz(1.826203306503199) q[2];
ry(1.6276612924503562) q[3];
rz(2.8110686850182773) q[3];
ry(2.1890295146320682e-05) q[4];
rz(-0.528936794270538) q[4];
ry(4.668180043410075e-06) q[5];
rz(-2.5763604287898163) q[5];
ry(-1.2933079552718331e-06) q[6];
rz(0.8483634485788425) q[6];
ry(3.141510477529229) q[7];
rz(-1.4608511004522642) q[7];
ry(2.4210958533951037) q[8];
rz(0.40216443239024807) q[8];
ry(-0.015558934376754431) q[9];
rz(3.1352496787291515) q[9];
ry(4.811623616873817e-06) q[10];
rz(-1.8832103733316063) q[10];
ry(-3.1415651620230767) q[11];
rz(2.345873635022399) q[11];
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
ry(-1.3575555911270738) q[0];
rz(3.141552757348305) q[0];
ry(-3.1415915477880865) q[1];
rz(1.768235053696238) q[1];
ry(-7.460393679446851e-07) q[2];
rz(0.5302094439864886) q[2];
ry(-3.141570377543266) q[3];
rz(-2.470586402507852) q[3];
ry(0.8501546272615847) q[4];
rz(0.073549934551143) q[4];
ry(3.1414845057607854) q[5];
rz(1.8945382730373552) q[5];
ry(3.115137869897657) q[6];
rz(2.3770497365673586) q[6];
ry(-1.227413419221313) q[7];
rz(2.2716412491174105) q[7];
ry(-1.206105332052232e-06) q[8];
rz(2.7465138850800903) q[8];
ry(2.2914961590505375e-06) q[9];
rz(-1.564034957170878) q[9];
ry(2.1531024906499927) q[10];
rz(-3.1415645955005433) q[10];
ry(-0.007028895191848683) q[11];
rz(3.134279144416383) q[11];
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
ry(-1.5425195879217688) q[0];
rz(0.9618381735731414) q[0];
ry(-3.14159099585431) q[1];
rz(1.5616326250252033) q[1];
ry(1.5708423541560728) q[2];
rz(-3.0452868199220267) q[2];
ry(0.12810180842634564) q[3];
rz(-2.044076865800704) q[3];
ry(0.10973408016708852) q[4];
rz(0.9158150789149745) q[4];
ry(-4.0747139351537953e-07) q[5];
rz(0.906155847124349) q[5];
ry(-0.000780976415028045) q[6];
rz(-1.8379141681046207) q[6];
ry(-3.1360969446158373) q[7];
rz(-1.9753901834005747) q[7];
ry(-1.5660263245332988) q[8];
rz(-0.6959483344255971) q[8];
ry(-1.5707972393268061) q[9];
rz(-1.2359337594534094) q[9];
ry(2.525070902286508) q[10];
rz(1.0841391395265156) q[10];
ry(1.5708355446625708) q[11];
rz(1.8448532931256985) q[11];
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
ry(1.570758090046131) q[0];
rz(-1.174211444756279) q[0];
ry(-3.141590682347389) q[1];
rz(-2.0084780525933965) q[1];
ry(3.1415852173271475) q[2];
rz(0.7927378723457177) q[2];
ry(3.1415922429945775) q[3];
rz(-2.5985296895636303) q[3];
ry(-3.1415638875975103) q[4];
rz(-2.208741376864678) q[4];
ry(0.0006115708455653601) q[5];
rz(-2.2618833908955063) q[5];
ry(3.1413745443051746) q[6];
rz(2.4019919486521792) q[6];
ry(2.5734171679488034e-06) q[7];
rz(-2.2349816084372622) q[7];
ry(-1.5708028387102357) q[8];
rz(-2.2912820659202433) q[8];
ry(-1.5708220914269924) q[9];
rz(0.9693771287164797) q[9];
ry(6.865405903779799e-06) q[10];
rz(-2.6549530100494723) q[10];
ry(-3.141583361186887) q[11];
rz(1.8450024074123954) q[11];
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
ry(3.141557648323052) q[0];
rz(1.9134059411106201) q[0];
ry(2.421042030853471e-06) q[1];
rz(1.7415127839328244) q[1];
ry(6.0958994714459525e-05) q[2];
rz(2.3911424603849603) q[2];
ry(-1.5319189119877654) q[3];
rz(3.034645007480391) q[3];
ry(0.11368273552453131) q[4];
rz(0.22606451624411136) q[4];
ry(-6.273030954566569e-07) q[5];
rz(2.0942739899327973) q[5];
ry(-3.1415825042204713) q[6];
rz(1.8007430357390746) q[6];
ry(-3.1415892291879355) q[7];
rz(-1.7495707831673393) q[7];
ry(1.7526646614030028e-05) q[8];
rz(0.6664544337841605) q[8];
ry(-3.1415925103353115) q[9];
rz(0.9880634056066555) q[9];
ry(-1.5707989738556205) q[10];
rz(3.0875619248567614) q[10];
ry(-1.5708073861850584) q[11];
rz(-1.5522509336324952) q[11];