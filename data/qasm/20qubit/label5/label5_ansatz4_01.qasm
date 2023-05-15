OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-1.570799926639089) q[0];
rz(-1.569857736632391) q[0];
ry(-1.5707927806274489) q[1];
rz(1.2461218113311263) q[1];
ry(-3.141592413553514) q[2];
rz(0.7926075732323046) q[2];
ry(3.097633853260437) q[3];
rz(-1.57070204209539) q[3];
ry(-1.5707916733384497) q[4];
rz(-1.5707958120971117) q[4];
ry(1.3119123792813546e-07) q[5];
rz(-1.7773291697607352) q[5];
ry(1.570795941266268) q[6];
rz(1.5707955921559744) q[6];
ry(-1.5707965916996764) q[7];
rz(1.1565657882947047) q[7];
ry(3.1415923471185936) q[8];
rz(2.7198465586350773) q[8];
ry(-3.141592635742838) q[9];
rz(1.8322250427971447) q[9];
ry(3.0854788679590683e-07) q[10];
rz(0.9986704166499231) q[10];
ry(-1.570795416893155) q[11];
rz(1.116646682976815) q[11];
ry(-1.5707963431626184) q[12];
rz(-2.9118553342286195) q[12];
ry(1.5707965539264839) q[13];
rz(-1.8995532343935566) q[13];
ry(-1.5707963381717667) q[14];
rz(-2.789567823622048) q[14];
ry(-2.6948307443466035) q[15];
rz(-1.570796428163561) q[15];
ry(-1.724331633656817e-05) q[16];
rz(-1.4933808060674443) q[16];
ry(-0.05133608110057519) q[17];
rz(-6.845223752768348e-05) q[17];
ry(-3.1315362689167583) q[18];
rz(-1.7383123435325594) q[18];
ry(-0.00018122665973763222) q[19];
rz(1.5193964645619682) q[19];
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
ry(1.577494288840417) q[0];
rz(-2.6437728513533987) q[0];
ry(-1.3598654702750732) q[1];
rz(2.45187551910023) q[1];
ry(1.5707980660489342) q[2];
rz(2.447659437717995) q[2];
ry(1.5707972158976782) q[3];
rz(1.5707960801673897) q[3];
ry(1.5707959165225691) q[4];
rz(-0.39441088566938337) q[4];
ry(1.5707961309015797) q[5];
rz(0.8499056085807741) q[5];
ry(-1.5707965211620023) q[6];
rz(2.7073151181141615) q[6];
ry(-2.9325328286578625) q[7];
rz(0.8327337903948298) q[7];
ry(3.1377589080802855) q[8];
rz(2.0884450633547282) q[8];
ry(1.15072575468389e-07) q[9];
rz(1.152760553834373) q[9];
ry(-1.5707965439839864) q[10];
rz(2.3786429455803564) q[10];
ry(1.6548577894681669e-06) q[11];
rz(-2.68737978026723) q[11];
ry(3.1106259768740703) q[12];
rz(1.777645601720745) q[12];
ry(-3.141589742640333) q[13];
rz(2.8128351939102108) q[13];
ry(-3.2501495039428245e-06) q[14];
rz(1.9478112491004438) q[14];
ry(-1.5709882892203517) q[15];
rz(1.5906278847615232) q[15];
ry(-1.5707963992250078) q[16];
rz(-1.5707969479246344) q[16];
ry(-1.5707975893754282) q[17];
rz(1.4541225536760722) q[17];
ry(1.5724693654057285) q[18];
rz(1.571998551110946) q[18];
ry(-1.5708036835241037) q[19];
rz(-3.1059948642214112) q[19];
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
ry(1.5708100413856032) q[0];
rz(-3.1415319396948926) q[0];
ry(-3.141530304271848) q[1];
rz(-2.4699182924520175) q[1];
ry(-1.6254647035924208) q[2];
rz(-0.07794387305368002) q[2];
ry(-1.570797122074163) q[3];
rz(0.08382758846955424) q[3];
ry(2.486071923563174e-09) q[4];
rz(-2.2456426894865706) q[4];
ry(3.141385147496043) q[5];
rz(1.8702638146502877) q[5];
ry(4.800583033012697e-06) q[6];
rz(2.8495234846126194) q[6];
ry(-4.991127625946007e-07) q[7];
rz(1.5228340737940813) q[7];
ry(-3.141592068449426) q[8];
rz(-2.636698369145613) q[8];
ry(1.5707969311290337) q[9];
rz(0.6233142102730868) q[9];
ry(1.5707950858518267) q[10];
rz(1.5707908812688387) q[10];
ry(0.805230423761417) q[11];
rz(-9.563467569595474e-05) q[11];
ry(-1.6358707250421292) q[12];
rz(-0.0014752227261380921) q[12];
ry(-1.6358087494719695) q[13];
rz(-3.1336285163361484) q[13];
ry(7.662798137743643e-07) q[14];
rz(-1.726645509084964) q[14];
ry(3.1331169727283523) q[15];
rz(0.02003640919935101) q[15];
ry(1.5707962346988833) q[16];
rz(1.570798376766907) q[16];
ry(-3.1415772987201698) q[17];
rz(-1.6874636817567472) q[17];
ry(-1.5706815143382193) q[18];
rz(1.3394420905519069) q[18];
ry(0.001122896397644667) q[19];
rz(-0.035440204636027914) q[19];
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
ry(1.5707958132257223) q[0];
rz(3.5383769735396524e-06) q[0];
ry(2.0393252304629073e-06) q[1];
rz(3.021956412625077) q[1];
ry(1.5708673643435338) q[2];
rz(-1.591264791893729) q[2];
ry(-1.5707965975869167) q[3];
rz(-1.570778787171883) q[3];
ry(-3.1415919362488256) q[4];
rz(2.0729608600022544) q[4];
ry(-3.1415920011697613) q[5];
rz(1.020357631176382) q[5];
ry(-2.0168878202150847e-07) q[6];
rz(-0.2338071759136353) q[6];
ry(3.3272817034912805e-07) q[7];
rz(0.47437478803397953) q[7];
ry(3.1292119118047124) q[8];
rz(3.1400368270435703) q[8];
ry(-3.1415923505134886) q[9];
rz(1.6049739934770422) q[9];
ry(-1.569347348995258) q[10];
rz(3.141577665208245) q[10];
ry(1.5708025283404716) q[11];
rz(2.334294293184989) q[11];
ry(-1.5707725811690096) q[12];
rz(1.5723641906542687) q[12];
ry(-0.2003548400024604) q[13];
rz(-0.9815059916227673) q[13];
ry(3.141586649002732) q[14];
rz(-2.7337338082355838) q[14];
ry(-0.06347559075434298) q[15];
rz(1.5705703188148208) q[15];
ry(-1.5707936369764317) q[16];
rz(1.5707973054932562) q[16];
ry(-1.5779786612946216) q[17];
rz(-1.9228723648673318) q[17];
ry(-2.237118371569036) q[18];
rz(1.5856185742472024) q[18];
ry(1.5707213642734528) q[19];
rz(-0.3928615768247015) q[19];
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
ry(-1.5707916009662197) q[0];
rz(1.3870843509926012) q[0];
ry(-1.771112574822098e-06) q[1];
rz(-2.125372541011332) q[1];
ry(2.8004486094869887) q[2];
rz(1.3805028956511478) q[2];
ry(1.5707968311518792) q[3];
rz(-2.9253380310368655) q[3];
ry(-0.9765280815888238) q[4];
rz(-1.691858427825097) q[4];
ry(-1.570796102493344) q[5];
rz(-2.9248739642099264) q[5];
ry(-3.1415868377314373) q[6];
rz(0.57539097621342) q[6];
ry(3.1415922393367337) q[7];
rz(-2.8140787298836583) q[7];
ry(-1.5707954721757504) q[8];
rz(-1.6918568316762472) q[8];
ry(-8.17583551970813e-07) q[9];
rz(0.08783724719408467) q[9];
ry(1.7053065150470144) q[10];
rz(-1.691847097379103) q[10];
ry(-0.0001757809153577483) q[11];
rz(1.876809336804552) q[11];
ry(2.603813422751508) q[12];
rz(-1.6918645126911829) q[12];
ry(-1.5135659907627996e-05) q[13];
rz(0.4725874758696902) q[13];
ry(2.0319521989073703e-07) q[14];
rz(0.04431453344159842) q[14];
ry(-1.5707953356762543) q[15];
rz(2.6408009855866816) q[15];
ry(-1.570795820363265) q[16];
rz(-0.12087173366995146) q[16];
ry(-3.1415806762738083) q[17];
rz(-2.4310148598886796) q[17];
ry(1.5818773961021566) q[18];
rz(-0.1191352086086301) q[18];
ry(-7.22295132097351e-05) q[19];
rz(-0.10823175810590818) q[19];