OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.30197753648266) q[0];
rz(-1.7343632905463595) q[0];
ry(-0.0003794483409358211) q[1];
rz(-0.07215713463697959) q[1];
ry(0.0016168282017154922) q[2];
rz(2.5328187921174123) q[2];
ry(0.019863208228485085) q[3];
rz(1.5961651276204873) q[3];
ry(-2.059225489647896) q[4];
rz(1.069112543222488) q[4];
ry(-3.141339381590948) q[5];
rz(-2.0658761997332267) q[5];
ry(-1.5651425714003508) q[6];
rz(-0.004231945872381799) q[6];
ry(0.003670542846838544) q[7];
rz(-1.8426626480077348) q[7];
ry(1.572084259104574) q[8];
rz(2.9742481443960505) q[8];
ry(-3.140452249672258) q[9];
rz(-0.35998234879290253) q[9];
ry(-8.574590594712106e-05) q[10];
rz(0.30122409004636896) q[10];
ry(-3.138310235196412) q[11];
rz(-2.243565807791108) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.013704942950270471) q[0];
rz(-1.3140906051539298) q[0];
ry(3.14044379677097) q[1];
rz(-2.648218342961801) q[1];
ry(0.00015053255131292076) q[2];
rz(-1.6033418150377718) q[2];
ry(-2.6984070569965755) q[3];
rz(2.2308739211838966) q[3];
ry(3.136193809333025) q[4];
rz(-2.479691522792025) q[4];
ry(1.5718130030511666) q[5];
rz(2.1916439209339407) q[5];
ry(1.607175232654036) q[6];
rz(1.0982908623452445) q[6];
ry(1.2218096039259285) q[7];
rz(0.0026850994076133) q[7];
ry(3.1105814816465425) q[8];
rz(2.440039105931558) q[8];
ry(0.0009253820429482762) q[9];
rz(2.5986505566859757) q[9];
ry(-3.1241571910239747) q[10];
rz(0.9870564737086936) q[10];
ry(0.3533693573901475) q[11];
rz(-0.003324364489967294) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.00012772977729945954) q[0];
rz(3.0411782697186234) q[0];
ry(-0.0003921528205759366) q[1];
rz(-2.4358073168270313) q[1];
ry(3.1399775779051975) q[2];
rz(-0.45634770356808213) q[2];
ry(-0.0005824256995914106) q[3];
rz(0.828366571906147) q[3];
ry(-0.0003636772739774585) q[4];
rz(-2.737167033187658) q[4];
ry(-3.1381755454228504) q[5];
rz(0.6204989271015418) q[5];
ry(-3.14076365139003) q[6];
rz(-1.1385851108653766) q[6];
ry(1.5693333073224451) q[7];
rz(1.5709003327041502) q[7];
ry(-0.0016421877181604927) q[8];
rz(-2.6855838533557383) q[8];
ry(1.578776183977717) q[9];
rz(-1.5647793271131833) q[9];
ry(3.14142104606914) q[10];
rz(-1.2358637623790427) q[10];
ry(-1.5717369928484624) q[11];
rz(1.568568763279779) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.7220028320841232) q[0];
rz(1.4324240403183026) q[0];
ry(0.001368091916661207) q[1];
rz(-0.13960286003225694) q[1];
ry(1.5703535578928456) q[2];
rz(0.008571827893723214) q[2];
ry(-0.02872901502914826) q[3];
rz(-0.9558517505988557) q[3];
ry(0.8400061069219183) q[4];
rz(-1.6316942503943368) q[4];
ry(-1.56534383161042) q[5];
rz(1.234326655338533) q[5];
ry(-3.1152697579463355) q[6];
rz(-0.36987814144612924) q[6];
ry(1.564458438652107) q[7];
rz(1.0108627203588911) q[7];
ry(2.420251206314634) q[8];
rz(-2.9580154628076487) q[8];
ry(-1.5726106175011445) q[9];
rz(-1.2412445715059481) q[9];
ry(3.1415364988058916) q[10];
rz(-2.8188114709203336) q[10];
ry(-1.5648486921770655) q[11];
rz(-0.32419643788556307) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1316909520552776) q[0];
rz(0.5872042654960676) q[0];
ry(-0.0003140308291876185) q[1];
rz(-2.759031488320671) q[1];
ry(1.5718674471638199) q[2];
rz(-2.298191576472679) q[2];
ry(-0.0005259919223998821) q[3];
rz(2.574457914324787) q[3];
ry(0.02211251579239981) q[4];
rz(0.7942481510343491) q[4];
ry(3.1258793271284038) q[5];
rz(-2.402187185064836) q[5];
ry(2.0351832188353463) q[6];
rz(0.15219837730947905) q[6];
ry(0.016406748807839655) q[7];
rz(-3.0806670496271362) q[7];
ry(-1.69566261465111) q[8];
rz(2.655226130179366) q[8];
ry(3.107484506503276) q[9];
rz(1.6135170681305455) q[9];
ry(-3.141163627712818) q[10];
rz(1.1124460476605753) q[10];
ry(0.03576738660114387) q[11];
rz(-1.3985629469828629) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.5799289510280143) q[0];
rz(3.1085414743696527) q[0];
ry(-3.1414820692353906) q[1];
rz(0.05691936322575583) q[1];
ry(-1.5973932361826926) q[2];
rz(-0.8590376808278731) q[2];
ry(3.134873644559912) q[3];
rz(-1.6004093377211723) q[3];
ry(1.5612170597050359) q[4];
rz(-0.1718774811551933) q[4];
ry(-3.1385077109374455) q[5];
rz(-0.5097364733652201) q[5];
ry(0.8405842148102671) q[6];
rz(-2.907859040825958) q[6];
ry(-0.0012827377345274726) q[7];
rz(-1.063404813503328) q[7];
ry(-0.8374503496277399) q[8];
rz(2.6515552105213294) q[8];
ry(0.0005415169079707738) q[9];
rz(1.8512222849428062) q[9];
ry(1.5705634263764254) q[10];
rz(0.0029140376511894097) q[10];
ry(-3.1406441568705925) q[11];
rz(1.4247142819753824) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.06743564091261543) q[0];
rz(-2.277088878299378) q[0];
ry(0.11021303602613847) q[1];
rz(1.6428630068077654) q[1];
ry(0.005703370629217553) q[2];
rz(0.5633043572766523) q[2];
ry(1.996723778765273) q[3];
rz(-1.5618918503146504) q[3];
ry(-3.082833702238899) q[4];
rz(-1.9622316361265193) q[4];
ry(2.5106398256891276) q[5];
rz(1.0494037786426793) q[5];
ry(0.008037186435010802) q[6];
rz(-1.849172557609835) q[6];
ry(0.9423436250944999) q[7];
rz(1.083650817531612) q[7];
ry(0.004613357303806965) q[8];
rz(-1.9527119502425825) q[8];
ry(-0.9910432625862742) q[9];
rz(2.0469985046570773) q[9];
ry(-1.5619214190309938) q[10];
rz(-1.8744824904714228) q[10];
ry(2.138791496115699) q[11];
rz(1.9813829586159226) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.139581445182443) q[0];
rz(-1.2141857436318277) q[0];
ry(-1.5652326680491067) q[1];
rz(0.02240136951825189) q[1];
ry(0.003264700351029326) q[2];
rz(1.3338417032541976) q[2];
ry(1.5581310615212012) q[3];
rz(1.332794600735764) q[3];
ry(0.0011428158662960541) q[4];
rz(-0.9801469234767497) q[4];
ry(-3.1237591986203412) q[5];
rz(1.4077767425354644) q[5];
ry(3.1341778639864017) q[6];
rz(-2.991569962989707) q[6];
ry(-0.0178401555444779) q[7];
rz(1.1601729030416095) q[7];
ry(-3.1359513797745437) q[8];
rz(-2.7080589348994475) q[8];
ry(0.023087683038010086) q[9];
rz(-0.9568318706805351) q[9];
ry(-3.130749301040159) q[10];
rz(0.125392314431521) q[10];
ry(0.02300029565876649) q[11];
rz(-3.077291181002319) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1395465904636937) q[0];
rz(2.840040305667563) q[0];
ry(1.5852431814288181) q[1];
rz(-0.28249678199801487) q[1];
ry(3.1399419613769095) q[2];
rz(2.79781203836568) q[2];
ry(3.024117533880679) q[3];
rz(3.048906175988395) q[3];
ry(-4.1749555522940796e-05) q[4];
rz(-1.7687509577155156) q[4];
ry(-3.127394851156919) q[5];
rz(-1.0866410752845361) q[5];
ry(-3.1414906405802263) q[6];
rz(1.926408941085791) q[6];
ry(-0.009498288600649156) q[7];
rz(2.597427435905257) q[7];
ry(-0.0036067439056239555) q[8];
rz(-2.72308820179098) q[8];
ry(3.1389512631834875) q[9];
rz(2.786094178637397) q[9];
ry(3.1378179612403594) q[10];
rz(-2.5632711430995205) q[10];
ry(-3.139940157873418) q[11];
rz(0.6019808308248494) q[11];