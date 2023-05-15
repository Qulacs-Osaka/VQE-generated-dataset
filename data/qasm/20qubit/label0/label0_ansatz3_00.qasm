OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(3.070315401874793e-05) q[0];
rz(-1.634163127849943) q[0];
ry(1.5735950962606684) q[1];
rz(-1.5683566084718608) q[1];
ry(-1.4662964169673938) q[2];
rz(0.0003165604150945356) q[2];
ry(2.9332370170856024) q[3];
rz(-1.4145359174999348) q[3];
ry(-0.007648714776054888) q[4];
rz(-1.5914440197341941) q[4];
ry(-2.736781699937563) q[5];
rz(-1.5712649241011443) q[5];
ry(1.363721356821807) q[6];
rz(-0.0038389958941450596) q[6];
ry(2.942419532613769) q[7];
rz(0.003535128753912712) q[7];
ry(3.1400231448980698) q[8];
rz(3.1382371208471005) q[8];
ry(-2.5461328566621617) q[9];
rz(0.00023622907870768156) q[9];
ry(-1.4891254780856036) q[10];
rz(-1.5716190668106664) q[10];
ry(1.5305713983135913) q[11];
rz(0.0005570949440944307) q[11];
ry(-2.9312113088234426) q[12];
rz(1.6000089701648303) q[12];
ry(-0.011156930183908514) q[13];
rz(-3.1072164647181517) q[13];
ry(-1.5806908252051242) q[14];
rz(1.3948857982662979) q[14];
ry(1.5707037482426163) q[15];
rz(0.9630714112123523) q[15];
ry(2.475044406711616) q[16];
rz(1.589740055093132) q[16];
ry(3.125462895872229) q[17];
rz(-1.573931685162452) q[17];
ry(0.19663056188288675) q[18];
rz(3.1401464423507655) q[18];
ry(-2.8342166773742257) q[19];
rz(0.006891657119280303) q[19];
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
ry(-1.3395634043873237) q[0];
rz(-1.571047206348729) q[0];
ry(0.15697827006225307) q[1];
rz(1.354506680946725) q[1];
ry(1.8724029024481972) q[2];
rz(-1.0361510212362415) q[2];
ry(-2.0888447358587228) q[3];
rz(3.1321490857007244) q[3];
ry(1.5700267636346243) q[4];
rz(1.12020052041124) q[4];
ry(-1.5704524344191935) q[5];
rz(1.6323452180016131) q[5];
ry(2.1447575347192647) q[6];
rz(-1.5703226579290033) q[6];
ry(-3.1326201380703713) q[7];
rz(1.4496194801249906) q[7];
ry(3.1159428409236294) q[8];
rz(1.5842553518819864) q[8];
ry(-2.062499634077623) q[9];
rz(-3.14110331978308) q[9];
ry(-1.5715065917088928) q[10];
rz(1.6497144817183056) q[10];
ry(-0.36725219972680717) q[11];
rz(-1.3973647370282452) q[11];
ry(3.138912779499988) q[12];
rz(-1.522416834053736) q[12];
ry(1.5708109402586121) q[13];
rz(-0.016298103521213244) q[13];
ry(2.814314765335523) q[14];
rz(-1.6452459463421) q[14];
ry(0.14972159734072843) q[15];
rz(-0.43445562908083524) q[15];
ry(3.0654294251224194) q[16];
rz(-1.5507044739745384) q[16];
ry(-2.5565323060051903) q[17];
rz(-1.576722025939228) q[17];
ry(-0.3122583284494487) q[18];
rz(-0.0033799079573757496) q[18];
ry(-1.1096285262226118) q[19];
rz(-1.572959293198183) q[19];
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
ry(-1.570738990036313) q[0];
rz(-2.7404501526510274) q[0];
ry(1.6488432112112072e-05) q[1];
rz(-2.927368825824879) q[1];
ry(-0.00019768626386186983) q[2];
rz(1.037531468115203) q[2];
ry(-3.126425547233625) q[3];
rz(3.131300241386306) q[3];
ry(2.6864845674993547) q[4];
rz(2.8585487183913667) q[4];
ry(-2.8841965671618075) q[5];
rz(-3.0967412590553534) q[5];
ry(3.128560168159085) q[6];
rz(1.5793993142081808) q[6];
ry(0.0017223059680575514) q[7];
rz(1.6884562249805557) q[7];
ry(2.939220649689886) q[8];
rz(-3.1368949667951145) q[8];
ry(0.6356084158631009) q[9];
rz(0.009467531129755903) q[9];
ry(-1.5746244873141082) q[10];
rz(1.5753162508578447) q[10];
ry(-3.1414715422660375) q[11];
rz(0.9336390584916279) q[11];
ry(-2.941461623890761) q[12];
rz(-1.5650447568656736) q[12];
ry(-3.1256374963974305) q[13];
rz(1.5524837101691293) q[13];
ry(-1.5760166856142062) q[14];
rz(-1.5496639783631092) q[14];
ry(3.1410682088457893) q[15];
rz(2.7222082399463394) q[15];
ry(-0.41372844898747824) q[16];
rz(1.5707922902777174) q[16];
ry(-0.0046275581218067074) q[17];
rz(-1.5651554883831775) q[17];
ry(0.09611426217303073) q[18];
rz(0.004493166058688658) q[18];
ry(1.5706335447796633) q[19];
rz(-0.4127592625738615) q[19];
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
ry(-3.1407160232778293) q[0];
rz(-1.1691263669289618) q[0];
ry(1.5710525261644728) q[1];
rz(-1.5682927215965339) q[1];
ry(-1.5721996204230857) q[2];
rz(-3.141174654702377) q[2];
ry(-0.5300068741712567) q[3];
rz(-3.140760958237773) q[3];
ry(1.5712681855675412) q[4];
rz(-1.5700854900683683) q[4];
ry(1.568446641437986) q[5];
rz(-1.57086170133034) q[5];
ry(-1.2514148783192836) q[6];
rz(0.0007516508225608044) q[6];
ry(1.541050687755508) q[7];
rz(-3.1413704406403933) q[7];
ry(-1.4785704259455987) q[8];
rz(3.1411470652090174) q[8];
ry(2.6585345091575463) q[9];
rz(-3.132837865037135) q[9];
ry(-1.6597227593126151) q[10];
rz(3.141268335492998) q[10];
ry(0.018438440899817915) q[11];
rz(-0.7601014321959554) q[11];
ry(-1.5679034734538977) q[12];
rz(3.1415592855848034) q[12];
ry(-1.5645808916650177) q[13];
rz(1.5707823708486877) q[13];
ry(1.5736486628001325) q[14];
rz(0.0003290899270984582) q[14];
ry(0.0020808457765681965) q[15];
rz(-1.9456567977300894) q[15];
ry(-1.5720346235819624) q[16];
rz(-0.3229070955262374) q[16];
ry(-1.5704046824583413) q[17];
rz(-0.4914410013948398) q[17];
ry(-1.5706594236829359) q[18];
rz(0.39970641798392076) q[18];
ry(-3.141373095253593) q[19];
rz(-2.0491805203604887) q[19];