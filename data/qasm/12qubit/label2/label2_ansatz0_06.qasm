OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.45115740035998264) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.30233267237964445) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.07743834552603815) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.36591319831692437) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.010620925021405514) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.07925006313803382) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.4432714827548583) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.29072064559594035) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.1594886773428097) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.001499861226612952) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.003929778553876199) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.5029403278625176) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.2863252397913978) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.2715939330947054) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.21572372922265384) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.6179912753969594) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(1.3447312416149786) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.3762117471306608) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.6401120402425047) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.5356620751983721) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.19405729199199198) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.0002492130920154712) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.0007402818398359446) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.44802524770940205) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.48749210626731304) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.1559010332195825) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.46133556071973736) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.7193636927225093) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.13355515746541485) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.48805991159682893) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.19340375881465488) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.0901018727252155) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.13503832736122937) q[11];
cx q[10],q[11];
rx(0.8881574054380835) q[0];
rz(-0.29002233763715773) q[0];
rx(-0.3337938881512181) q[1];
rz(-0.09829403665066391) q[1];
rx(-0.8358377310091953) q[2];
rz(0.03719149978782745) q[2];
rx(0.3248035609864269) q[3];
rz(-0.33750545520822356) q[3];
rx(0.20088772703779273) q[4];
rz(-0.06541346561012579) q[4];
rx(0.06415403002287302) q[5];
rz(-0.1889857491026137) q[5];
rx(0.853762526497409) q[6];
rz(-0.12119184591242227) q[6];
rx(-0.7511910792765941) q[7];
rz(-0.6203104084439768) q[7];
rx(-0.13376212592164052) q[8];
rz(-0.5503836773673263) q[8];
rx(-0.46985951096339995) q[9];
rz(-0.1307467662399363) q[9];
rx(-1.3252708824718713) q[10];
rz(0.029915088394915284) q[10];
rx(0.07696699808833538) q[11];
rz(0.140711644158747) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.5022044768473423) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.34547968425396547) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.7236080818444393) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.18569624375351904) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.7064836843720163) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.13703243114131364) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.10638907509606364) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.5443393437800522) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.32854609409573277) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.0028876254998267286) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(0.004518478644707914) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.007701689100788099) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.02134139158693505) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.018808762635457877) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.07877186909361651) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-1.0291819842518608) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.6800669310943851) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.4164814851860955) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.9779154361426282) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.7068101730804043) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.20138517941014136) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.0009273075057819894) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.00015630537659231482) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.00039094674885830916) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.12738467060968311) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.25530114472235077) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.16136786971703462) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.2868464964665723) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-1.3548376925473609) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.8159414200290026) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.5141423706723695) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.13707233141899333) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.38928330871570416) q[11];
cx q[10],q[11];
rx(0.7112050196867473) q[0];
rz(-0.3559748651292337) q[0];
rx(-0.27513110594320983) q[1];
rz(-0.6467537255331325) q[1];
rx(-1.0782457844771134) q[2];
rz(-0.2759800117487908) q[2];
rx(0.8449268009831575) q[3];
rz(0.1924152416035298) q[3];
rx(0.20578565119074058) q[4];
rz(-0.0349848081881179) q[4];
rx(-0.21137592902743496) q[5];
rz(0.7510429908789277) q[5];
rx(0.5696761998960768) q[6];
rz(-0.6750569026043244) q[6];
rx(-0.24897473414124402) q[7];
rz(-0.5875278551253119) q[7];
rx(-0.7318875452626785) q[8];
rz(-0.3984323874070389) q[8];
rx(-0.06720175098535389) q[9];
rz(-0.13533879562238244) q[9];
rx(-0.6556673663877829) q[10];
rz(0.5090321789298564) q[10];
rx(0.20335123927842333) q[11];
rz(-0.55775711677876) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.11724235149103703) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.5702526896093895) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.6690092802647101) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.01754461986003067) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.04056127678016536) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2364429154881888) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.42615769184666824) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.7879572816428816) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.14555520860949114) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.0029038940494055137) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(0.0003659903043280851) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.0005004701612426561) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.011418658721704392) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.003336900184044628) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.056159978749802934) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.7268997309477278) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.8477182228310206) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.04594763700976722) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.6030159098943952) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.2205935706341602) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.046584310133461905) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.0013549626319577918) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-5.9377907733281215e-06) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.0006305686095196877) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.8194970429623761) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.1007000019029978) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.4213264290127505) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.9500289114922702) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-1.172739328205297) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.36206749654596854) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.6021063970442462) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.4885121098930017) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.80142468733363) q[11];
cx q[10],q[11];
rx(0.5373452009731263) q[0];
rz(-0.19943918929840895) q[0];
rx(-0.5265090903860561) q[1];
rz(-0.7858736580294711) q[1];
rx(-1.7069217401754868) q[2];
rz(-0.5056036806362235) q[2];
rx(1.2512957746086522) q[3];
rz(-0.6383911714247864) q[3];
rx(0.28581036468346654) q[4];
rz(0.6119596496017445) q[4];
rx(-0.26574152423967434) q[5];
rz(0.30358380467479684) q[5];
rx(0.41065940886504754) q[6];
rz(0.35378912093663906) q[6];
rx(-0.6748942268999176) q[7];
rz(0.11708387813419324) q[7];
rx(-0.998204940498318) q[8];
rz(-0.29771704589775383) q[8];
rx(1.0847929651409125) q[9];
rz(-0.1543968061947167) q[9];
rx(-0.43433354356257436) q[10];
rz(-0.5594871413549596) q[10];
rx(0.7911236860432608) q[11];
rz(-0.27319753985487666) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.29933868021778665) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.38737195783588935) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0016589066431669985) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.2767838027197133) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.7464800276528073) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.52466829583013) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.3868864609302449) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.10311408570443345) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.3849761542759797) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.0023149370978634066) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.006559873893229913) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.000346626785887308) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.03294108326429079) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.021060414418961563) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.04354629478999444) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-1.0094809031335057) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.7233772492876314) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.30901719162392255) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.4692042654576372) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.3583691294900267) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.1723569280928542) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.9889814497721816) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.34441087473798565) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.5482526339576684) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.0013833762083618562) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.00016747799470297173) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.0006166904805326269) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.6145526369592054) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-1.2965505545929132) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.6681172105122655) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.7382261047913828) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.20782195586148153) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.23974209224776172) q[11];
cx q[10],q[11];
rx(0.25038419850374843) q[0];
rz(-0.24450334734049928) q[0];
rx(-1.0087358061328693) q[1];
rz(-0.7625110148520216) q[1];
rx(-1.104928428458553) q[2];
rz(-0.5368605723907015) q[2];
rx(1.1066131069792364) q[3];
rz(-1.1133700618161522) q[3];
rx(0.0515603727470154) q[4];
rz(-0.5434317822153882) q[4];
rx(-0.07408989511917592) q[5];
rz(0.020431808709672438) q[5];
rx(0.3779464769110014) q[6];
rz(0.34556052320187775) q[6];
rx(-0.4246840996449406) q[7];
rz(-0.04247503986034015) q[7];
rx(-0.9421779361907532) q[8];
rz(0.0002831580997696872) q[8];
rx(1.5473160662214591) q[9];
rz(-0.14431830131095574) q[9];
rx(0.0839726525416193) q[10];
rz(-0.1246391232535233) q[10];
rx(0.19477673481295793) q[11];
rz(0.014456502796135142) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.09457485272163019) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.8000027714947652) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.22479432612382597) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(1.3846264726817232) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-1.1306204046166457) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.5866149227222485) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.2764390475535464) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.0841104670722679) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.09227970779832806) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.4394547916178667) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.45808097376081025) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.6696865210932986) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.0005700954622776097) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.0012743422507067645) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(4.240464930129656e-05) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.60714326919461) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.590447374386362) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.9109781998063472) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.0014658384465460947) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.0003053805967099677) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.0006264963619346549) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-1.1041179722761572) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.03713354299483576) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.286320220567622) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.0011237384948784425) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(8.016456183476163e-05) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.0001418569778533431) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(1.0052490814894495) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-0.439713127477916) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.26457638782099174) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.5850534153350916) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.6451047618299667) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.7238109333137018) q[11];
cx q[10],q[11];
rx(0.36655123184452754) q[0];
rz(-0.25749418881193814) q[0];
rx(-1.3878145714308963) q[1];
rz(0.41899372453920386) q[1];
rx(-0.8886903400724904) q[2];
rz(-0.729149048678671) q[2];
rx(1.03392851195512) q[3];
rz(-0.7365727268963893) q[3];
rx(0.6693760341735567) q[4];
rz(0.12794067150932326) q[4];
rx(-0.6682678175278762) q[5];
rz(0.1329575350535) q[5];
rx(0.03686876499197473) q[6];
rz(0.2545366424646459) q[6];
rx(-0.2763562658128612) q[7];
rz(0.038909609934421927) q[7];
rx(-1.2402870050052293) q[8];
rz(-0.01831351376804079) q[8];
rx(1.481571953020146) q[9];
rz(-0.5110753554064091) q[9];
rx(0.11255692170663696) q[10];
rz(0.25488679051854696) q[10];
rx(0.7789020918422891) q[11];
rz(0.3919864485661972) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.01218210986501013) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.9246963981560734) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.0352219512928999) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(1.5470495085113554) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-1.8409232543812915) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.3208733660002112) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.000447528588991051) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0012493325138638778) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.002233471009741668) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.6449798441026664) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-1.5573844672459478) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.04955290792041696) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-7.961699166639398e-06) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.00108209405162211) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.0011332105317113752) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.3678821601156517) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.446104773512957) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(1.1602918866005343) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.0005665422311964015) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.00023654793292983761) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.0013609017402339103) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-1.0554543301156067) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.37365935281912777) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.49948011185515934) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.09982562779997704) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.1364580887103159) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.10780409036652426) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(1.2305943114026223) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-1.3777457243215994) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.6589074225898338) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.25348984899581967) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.22855524379297368) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.505902142972073) q[11];
cx q[10],q[11];
rx(0.45101444266475205) q[0];
rz(-0.7887256225090952) q[0];
rx(-1.2616757689347642) q[1];
rz(0.722854984869212) q[1];
rx(0.19643785394280822) q[2];
rz(-0.4869720581124062) q[2];
rx(0.5792460492047501) q[3];
rz(-0.05801975168693511) q[3];
rx(0.42835959468973894) q[4];
rz(0.5303448102720402) q[4];
rx(0.13873085937631396) q[5];
rz(0.37017752045321356) q[5];
rx(0.37339901493622396) q[6];
rz(0.27526468149926303) q[6];
rx(0.03110183757486647) q[7];
rz(-0.4660457099999032) q[7];
rx(-1.2642169071404916) q[8];
rz(-0.42327793460503593) q[8];
rx(1.3564736167904734) q[9];
rz(-0.3724624190205922) q[9];
rx(0.07012810848615493) q[10];
rz(-0.7641363094349126) q[10];
rx(0.3378769719196624) q[11];
rz(0.40564976809545733) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.3223892279270929) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.16291127665089386) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.7357107427971646) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(2.0814053968386146) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-2.1461622080039344) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.1678539680429332) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.002332507730549793) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.00015174137004245484) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(8.791916423098765e-05) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.44180534604476995) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-1.708298934005863) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.7151321831460516) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.0005815099267808345) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.00032487643582690396) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.0004937176776272659) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.2280041439100075) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.04245980242979228) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(1.482170902584007) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.19351011202872534) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.3782119850740919) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(1.2431897096507254) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.013311907248231754) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.016462279749191962) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.01048882518571829) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.013957439639204693) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.002551994819536816) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.0018724908045151642) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.8603574008908532) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-1.465060507390113) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-1.259799327961962) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.08414657252192143) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.05438575477284641) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.07255424677905348) q[11];
cx q[10],q[11];
rx(0.3226229379296933) q[0];
rz(-0.53414421626766) q[0];
rx(-0.7020464502730379) q[1];
rz(0.3560887360338766) q[1];
rx(-1.4777743243351882) q[2];
rz(-1.0405941703051071) q[2];
rx(1.0602131968647837) q[3];
rz(1.1249365683143193) q[3];
rx(0.10691226167702697) q[4];
rz(0.9241506409108698) q[4];
rx(0.368908281796646) q[5];
rz(0.3856477712862481) q[5];
rx(0.10401416406227289) q[6];
rz(0.061809000887318735) q[6];
rx(-0.5614163247641696) q[7];
rz(-1.3100822080950663) q[7];
rx(-0.2652684538606157) q[8];
rz(0.37680957645959695) q[8];
rx(0.7924368751954567) q[9];
rz(0.4527718679977951) q[9];
rx(-0.29747142333402055) q[10];
rz(-0.6198414662565322) q[10];
rx(0.17996136887646993) q[11];
rz(0.3266309591543928) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.2757273900425739) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.40747667189940423) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04675271354483264) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(1.8781980916675622) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-1.275285160409008) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.7714979088794262) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.0009171799739424744) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.0009744840824189955) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.00034819554933637823) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.6634657510478356) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-1.22419554519597) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.33371837926586645) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.00018057558003956888) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.7458576949306179) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.027395010409642487) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.003748658699794956) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.011695980839672212) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.0032617192816405856) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.7191503497283734) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.32953403729450087) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(1.9125021740446204) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.007597892388580248) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.0005290026638776719) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.005617344617915314) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.02003483920387249) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.005177778124939874) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.02489536139468919) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(1.0057149411678965) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-1.3254312199548188) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.9793337019785809) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-1.042271847909636) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.473813230430726) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.3914046570592105) q[11];
cx q[10],q[11];
rx(0.505157660168523) q[0];
rz(0.10096834161548525) q[0];
rx(-0.9243189076010524) q[1];
rz(-0.1718346219288012) q[1];
rx(-1.6051748350988744) q[2];
rz(-0.10482526143298108) q[2];
rx(0.18121896401467014) q[3];
rz(1.0022721215377521) q[3];
rx(0.07302241196077117) q[4];
rz(0.002444728072855507) q[4];
rx(-1.4562758289009805) q[5];
rz(-0.8648932409005764) q[5];
rx(-0.017081288415966133) q[6];
rz(0.2827984392532086) q[6];
rx(-0.149302301822114) q[7];
rz(-0.44733276218197304) q[7];
rx(-0.4369841717519051) q[8];
rz(1.0456109934175053) q[8];
rx(0.5994532132999731) q[9];
rz(0.7860403823430555) q[9];
rx(-0.981643018635075) q[10];
rz(0.16211876750522608) q[10];
rx(0.31541101925902887) q[11];
rz(0.35617087140576353) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.33746312873253664) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1751302944245243) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.6505623939717713) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(2.0744598164752897) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-1.141834065547453) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.7317552695450443) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.00446043438075144) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.48625779686180226) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06144418360482251) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.07343514467950499) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.5010948172434879) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.062459213117389015) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.15379219390792356) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-1.2843019815009875) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.2750850549687643) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.05277846507827267) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.047587848694325176) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.06213027842532003) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.7316787141911602) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.8338712311696685) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.7166080105173486) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.0610383204189375) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.06171002058182916) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.0564127519234958) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.8120084494558771) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.7649904775256624) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.7652099080786264) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.05909524510800463) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-0.061265790067408926) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.06800978961628457) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(2.3767081209572116) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.8189918295808679) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.817283229615955) q[11];
cx q[10],q[11];
rx(0.2080460253022791) q[0];
rz(0.26149301024593957) q[0];
rx(-0.8636759826761129) q[1];
rz(-0.4238923503506177) q[1];
rx(-0.3412133680027717) q[2];
rz(0.1532428006701696) q[2];
rx(1.2581606542024875) q[3];
rz(-0.04447772537054663) q[3];
rx(0.822707385224401) q[4];
rz(-0.049437622849620236) q[4];
rx(-0.046532845310725414) q[5];
rz(-0.04559537235740962) q[5];
rx(0.08479461168399838) q[6];
rz(-0.10982306698645769) q[6];
rx(0.0800672808283466) q[7];
rz(-0.0747198284389899) q[7];
rx(0.024173829381356683) q[8];
rz(-0.08028536050719402) q[8];
rx(0.13193268885547055) q[9];
rz(-0.09885319963026198) q[9];
rx(0.1532788821870586) q[10];
rz(-0.17030962666947339) q[10];
rx(0.1691593219734651) q[11];
rz(0.042663195844461384) q[11];