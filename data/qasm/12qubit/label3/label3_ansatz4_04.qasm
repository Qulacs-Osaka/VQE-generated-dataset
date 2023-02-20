OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.5707999132507044) q[0];
rz(1.692549593190817) q[0];
ry(-3.1415895403274043) q[1];
rz(-0.939172561328177) q[1];
ry(-1.5707912790544216) q[2];
rz(2.6849182584579694) q[2];
ry(-1.5707913710265364) q[3];
rz(0.7839434857378745) q[3];
ry(-3.1415884873851163) q[4];
rz(0.21089910808116663) q[4];
ry(1.570796399259371) q[5];
rz(-1.5686711587156064) q[5];
ry(1.5708215790177062) q[6];
rz(-3.0868816720536594) q[6];
ry(3.829697448054504e-08) q[7];
rz(-3.0579884110511992) q[7];
ry(-3.1415922809891117) q[8];
rz(0.915161349147753) q[8];
ry(0.33545031198950503) q[9];
rz(-1.5698329006399792) q[9];
ry(3.0401922713435714) q[10];
rz(-1.0149334828233068) q[10];
ry(1.5708082200216935) q[11];
rz(-1.5707963657612911) q[11];
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
ry(-3.1413198832434475) q[0];
rz(-3.0198353452185813) q[0];
ry(1.5704314352727895) q[1];
rz(2.4029287977581784) q[1];
ry(3.141592222960295) q[2];
rz(2.624023491662679) q[2];
ry(-0.3452896504172255) q[3];
rz(0.5694436618830246) q[3];
ry(1.5708085368479539) q[4];
rz(-3.141592594264748) q[4];
ry(1.5707265077321635) q[5];
rz(-0.015344608471083376) q[5];
ry(3.1415910950559067) q[6];
rz(1.6254861746356326) q[6];
ry(-1.6766849146311464) q[7];
rz(-0.9106115270475278) q[7];
ry(-2.648977768674248e-07) q[8];
rz(-1.643200926249257) q[8];
ry(-1.5680134296600023) q[9];
rz(-0.8205808147846925) q[9];
ry(6.794872407657593e-07) q[10];
rz(1.0162285806747708) q[10];
ry(1.5720987722764859) q[11];
rz(-3.1415709333935826) q[11];
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
ry(-0.08044592815115068) q[0];
rz(-1.5707910247357715) q[0];
ry(1.9447736547595517e-07) q[1];
rz(-2.184462078079453) q[1];
ry(-7.194171586782804e-05) q[2];
rz(1.6316898214423832) q[2];
ry(-0.004928970957800871) q[3];
rz(-2.953404782928806) q[3];
ry(1.5416653270915934) q[4];
rz(1.113860941259759e-07) q[4];
ry(1.5707961615949246) q[5];
rz(1.5707952935454426) q[5];
ry(-0.009005088549786322) q[6];
rz(1.5708178846168797) q[6];
ry(5.820326973268213e-09) q[7];
rz(-2.5230437763086946) q[7];
ry(-3.1415926326770576) q[8];
rz(-0.09436841325033536) q[8];
ry(0.012922969878708558) q[9];
rz(-3.1004965852594264) q[9];
ry(0.018739268370172958) q[10];
rz(-1.8723653843949875) q[10];
ry(-0.7620756936025908) q[11];
rz(3.141562148118128) q[11];
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
ry(1.5708174012132683) q[0];
rz(3.1413216955654994) q[0];
ry(-3.1412252021813214) q[1];
rz(2.3836669947290474) q[1];
ry(-1.5707963251383898) q[2];
rz(-1.1650375673077917e-05) q[2];
ry(-1.5707948316215303) q[3];
rz(-3.8181787687108226e-07) q[3];
ry(1.570802946446384) q[4];
rz(1.568541052902109) q[4];
ry(-1.573513678036377) q[5];
rz(-1.5707338146741454) q[5];
ry(1.57078456576098) q[6];
rz(-1.573579443738612) q[6];
ry(3.141592435886268) q[7];
rz(2.8495406922535906) q[7];
ry(1.570796447463311) q[8];
rz(-1.5707888546666426) q[8];
ry(-1.5737197748664702) q[9];
rz(-0.53577404657675) q[9];
ry(1.5707960063898365) q[10];
rz(-1.4269122718957303e-06) q[10];
ry(1.2841119827494767) q[11];
rz(-1.570796924832501) q[11];
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
ry(1.570794902982005) q[0];
rz(-1.5707952154830016) q[0];
ry(-2.9484681600643037e-07) q[1];
rz(-2.141997518659395) q[1];
ry(-3.0321972164707103) q[2];
rz(-1.5707917612189337) q[2];
ry(1.5707935720874628) q[3];
rz(2.890676270573826) q[3];
ry(0.08926417802885127) q[4];
rz(1.574981424468457) q[4];
ry(-0.0884693792081448) q[5];
rz(1.5671868308149068) q[5];
ry(1.5709394313986647) q[6];
rz(0.018014719570317687) q[6];
ry(1.5707962981001602) q[7];
rz(9.565642500319882e-09) q[7];
ry(0.011710295593403036) q[8];
rz(2.3886973841712273) q[8];
ry(3.141592426417485) q[9];
rz(1.0377132532720217) q[9];
ry(-1.5707961245934392) q[10];
rz(1.888606704823746) q[10];
ry(1.5707965126273373) q[11];
rz(0.09299734884911033) q[11];
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
ry(1.46588386396758) q[0];
rz(-0.3431640760864285) q[0];
ry(-1.5707520074044752) q[1];
rz(2.5728511164302224) q[1];
ry(-1.715559872397702) q[2];
rz(-1.570771029130508) q[2];
ry(1.5707969588548585) q[3];
rz(3.1415925891577583) q[3];
ry(-0.014162082843021116) q[4];
rz(-0.5036688800485338) q[4];
ry(0.017733651339664647) q[5];
rz(-2.669600680821655) q[5];
ry(1.570795542413107) q[6];
rz(2.1490586816212) q[6];
ry(1.57079666642165) q[7];
rz(0.9430219330370244) q[7];
ry(-1.3225273611780662e-07) q[8];
rz(0.7528880702721372) q[8];
ry(1.5707964933203613) q[9];
rz(-1.5707981710692793) q[9];
ry(2.8372632818477803) q[10];
rz(0.3041090120838686) q[10];
ry(-1.5997076691867882) q[11];
rz(2.841113675653648) q[11];
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
ry(-1.3751604314826693e-05) q[0];
rz(-2.798425842188471) q[0];
ry(-3.141580235657127) q[1];
rz(-2.1395497247524506) q[1];
ry(-1.570796355567289) q[2];
rz(-1.5707998682389943) q[2];
ry(1.5708014003136823) q[3];
rz(3.1415755372518688) q[3];
ry(2.9590419803327e-07) q[4];
rz(-0.3075965661087971) q[4];
ry(3.1415918705642922) q[5];
rz(2.5033506025715457) q[5];
ry(-1.8522618994154527e-08) q[6];
rz(-2.6687498804713834) q[6];
ry(3.1415926140854276) q[7];
rz(-2.198573456810405) q[7];
ry(-1.570796341739332) q[8];
rz(-1.7641655275588546) q[8];
ry(-1.570796355240385) q[9];
rz(3.1415909218454114) q[9];
ry(1.5707962965731905) q[10];
rz(-0.24492614443384425) q[10];
ry(-1.5707968656003999) q[11];
rz(1.1558833695822184) q[11];
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
ry(-1.5707935466522374) q[0];
rz(-1.9011206641123097) q[0];
ry(1.5707963830869858) q[1];
rz(-1.9011238537190291) q[1];
ry(-1.3198328486092465) q[2];
rz(-0.3303231073947588) q[2];
ry(-1.5707991344083974) q[3];
rz(1.0957079462997197) q[3];
ry(-5.193871714936905e-06) q[4];
rz(2.051345590938192) q[4];
ry(1.390646231933335e-05) q[5];
rz(-0.7908124555327225) q[5];
ry(-4.310151670843153e-06) q[6];
rz(1.7638254654198349) q[6];
ry(-1.5707899784435133) q[7];
rz(-1.8975118579835872) q[7];
ry(-4.615100970509048e-06) q[8];
rz(1.4374623101807718) q[8];
ry(1.5708032203680509) q[9];
rz(-1.897499914094615) q[9];
ry(4.623914322507869e-06) q[10];
rz(-1.6525733999594454) q[10];
ry(-1.7251371366271957e-05) q[11];
rz(0.08820941500426563) q[11];