OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.5708028180885976) q[0];
rz(3.2522359587616734e-06) q[0];
ry(1.5707969853762038) q[1];
rz(1.5707951708207601) q[1];
ry(3.141592419512849) q[2];
rz(-1.5570237763477213) q[2];
ry(-1.5707968364333806) q[3];
rz(1.5707963688638633) q[3];
ry(-1.4135042629580479) q[4];
rz(0.876771022085673) q[4];
ry(-3.141591878269318) q[5];
rz(-0.7245550435150954) q[5];
ry(3.141591640379218) q[6];
rz(0.7042330075047929) q[6];
ry(-6.674151418195606e-07) q[7];
rz(1.199220388712586) q[7];
ry(-3.1415902434807235) q[8];
rz(2.4185801327646046) q[8];
ry(-3.1415923508297423) q[9];
rz(1.424127660227247) q[9];
ry(3.141592621955183) q[10];
rz(0.2956470100302655) q[10];
ry(-3.333212337253144e-07) q[11];
rz(-2.7028343767921803) q[11];
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
ry(-2.6806352553275388) q[0];
rz(-0.7750828258575311) q[0];
ry(-1.570797233905631) q[1];
rz(2.6233595450460085) q[1];
ry(1.5707960219939714) q[2];
rz(1.570794491596181) q[2];
ry(-1.5707960559329806) q[3];
rz(1.5135242894113707) q[3];
ry(-3.1415893646921016) q[4];
rz(1.1980504178576235) q[4];
ry(-1.7058608570152956e-07) q[5];
rz(-2.2844491047551174) q[5];
ry(-0.018417389515145877) q[6];
rz(0.807295375129855) q[6];
ry(-3.141592023533204) q[7];
rz(-0.19312453410008157) q[7];
ry(-2.9991331064461266) q[8];
rz(-0.605523505058402) q[8];
ry(-1.5707972015869276) q[9];
rz(-3.1415925265331444) q[9];
ry(1.5338703626968964) q[10];
rz(2.5098037044781694) q[10];
ry(-1.5707944218273375) q[11];
rz(1.2838184656933453) q[11];
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
ry(3.141532806757803) q[0];
rz(2.3665069706552084) q[0];
ry(1.570796441403988) q[1];
rz(-3.1415578788476193) q[1];
ry(1.5707971826847116) q[2];
rz(1.1620363558023543) q[2];
ry(1.570796380809428) q[3];
rz(-4.6201567973572936e-08) q[3];
ry(-0.0003496843118782067) q[4];
rz(-1.8920754949566243) q[4];
ry(3.141592572333453) q[5];
rz(0.34991582116288694) q[5];
ry(3.14159258785608) q[6];
rz(-0.7632683146305288) q[6];
ry(3.1415922811852464) q[7];
rz(0.04401373894984495) q[7];
ry(6.054090193946753e-07) q[8];
rz(0.6055288754558725) q[8];
ry(1.0026277768301979) q[9];
rz(1.5707966736254877) q[9];
ry(-3.7317561877490856e-07) q[10];
rz(-0.14261165621491578) q[10];
ry(5.572591277314132e-10) q[11];
rz(3.0350805331394786) q[11];
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
ry(1.5707542700434907) q[0];
rz(-2.14241436567129) q[0];
ry(0.09081446226320633) q[1];
rz(0.42719663815227626) q[1];
ry(-3.1415921347934472) q[2];
rz(-2.8292654142491704) q[2];
ry(1.4799812253595568) q[3];
rz(-0.70574865611459) q[3];
ry(1.5707982994014964) q[4];
rz(1.5707553995690215) q[4];
ry(-3.1415917863495424) q[5];
rz(1.2241658614587063) q[5];
ry(1.570794985095299) q[6];
rz(-1.8386606898826807) q[6];
ry(-9.585993149123148e-08) q[7];
rz(-3.052059684287119) q[7];
ry(-2.5397992884342164) q[8];
rz(-3.141591027095652) q[8];
ry(-1.5695533694610315) q[9];
rz(-1.570794633401217) q[9];
ry(3.139814802694395) q[10];
rz(0.7963958182613613) q[10];
ry(3.1415902025309546) q[11];
rz(1.1773378218026782) q[11];
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
ry(-3.14159227493536) q[0];
rz(0.9991772227545771) q[0];
ry(-4.06585224540757e-07) q[1];
rz(-0.47732095150725445) q[1];
ry(3.141591831105983) q[2];
rz(2.860050575685042) q[2];
ry(3.1415920730481472) q[3];
rz(2.4358437620340188) q[3];
ry(2.409053682704797) q[4];
rz(-2.4673787310192097e-05) q[4];
ry(-1.570799099879551) q[5];
rz(-3.1415850330829853) q[5];
ry(3.1415924550445786) q[6];
rz(1.3354326023577396) q[6];
ry(1.5707975959753382) q[7];
rz(8.523520280689922e-07) q[7];
ry(-1.5707986764488748) q[8];
rz(1.57079346619878) q[8];
ry(1.4406762180501946) q[9];
rz(2.232909358424865) q[9];
ry(-1.5707974197328018) q[10];
rz(-3.1415480614795843) q[10];
ry(0.13016378178344998) q[11];
rz(1.5679192628098428) q[11];
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
ry(1.5707992333703311) q[0];
rz(-3.141569030796011) q[0];
ry(3.1413894692810755) q[1];
rz(-0.8310707024595337) q[1];
ry(-1.570794035922911) q[2];
rz(-2.0079702594072444) q[2];
ry(1.6575588428487222) q[3];
rz(-1.7591053406868777) q[3];
ry(2.9192311502138826) q[4];
rz(-1.9907695474621718) q[4];
ry(-2.5400826896176474) q[5];
rz(-3.141585597946115) q[5];
ry(-1.5647975827743092) q[6];
rz(2.3235556677794786) q[6];
ry(-0.969283680352401) q[7];
rz(3.141591337257729) q[7];
ry(-1.5587470805509285) q[8];
rz(3.0413304304706363) q[8];
ry(1.424034564024973e-07) q[9];
rz(0.9086823786003686) q[9];
ry(-1.5707976122564586) q[10];
rz(-3.1415899904687734) q[10];
ry(0.0031048606783411915) q[11];
rz(-2.2997909339906624) q[11];
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
ry(-3.068981461556137) q[0];
rz(0.35352411773029185) q[0];
ry(1.3070775717144113e-06) q[1];
rz(-0.7898142106831187) q[1];
ry(-8.109503211171258e-08) q[2];
rz(-1.8612606615083243) q[2];
ry(-1.6218062103173633e-06) q[3];
rz(0.18830913564008525) q[3];
ry(3.1415926322001684) q[4];
rz(2.721648094773218) q[4];
ry(1.5707963027419245) q[5];
rz(2.983285093412207) q[5];
ry(1.5707967931925582) q[6];
rz(3.1415921311905537) q[6];
ry(-1.570796036356277) q[7];
rz(-0.47083225042933213) q[7];
ry(3.1415925562837796) q[8];
rz(-1.6710593956394826) q[8];
ry(1.570799008704598) q[9];
rz(-3.1415918476391056) q[9];
ry(1.5707966840398515) q[10];
rz(-2.0253005730294715) q[10];
ry(-3.141590490963272) q[11];
rz(-0.7318371666037924) q[11];
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
ry(-6.777837047167168e-08) q[0];
rz(0.6235670709551151) q[0];
ry(-1.5707963612590252) q[1];
rz(1.741015382487681) q[1];
ry(1.699581535952177e-06) q[2];
rz(-1.4368851370499256) q[2];
ry(1.570794697558962) q[3];
rz(1.741015780680491) q[3];
ry(1.570798042429564) q[4];
rz(-0.5937323894879055) q[4];
ry(9.584828778954253e-07) q[5];
rz(1.3371500947470318) q[5];
ry(-1.5707960457171546) q[6];
rz(2.5478670095789724) q[6];
ry(-3.1415917454070423) q[7];
rz(-0.8627864045084541) q[7];
ry(1.5707945973935473) q[8];
rz(0.9770713428470907) q[8];
ry(1.5707959653007055) q[9];
rz(0.1613116179953176) q[9];
ry(1.7913755678833354e-06) q[10];
rz(-0.13922133383627128) q[10];
ry(-1.5707923050165524) q[11];
rz(0.1613089718412999) q[11];