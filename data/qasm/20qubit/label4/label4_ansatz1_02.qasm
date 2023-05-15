OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(2.720220429231352) q[0];
rz(-2.043740510894618) q[0];
ry(-3.141061581653302) q[1];
rz(-0.4321977079194355) q[1];
ry(-1.5252475460611845) q[2];
rz(-1.5703968093305782) q[2];
ry(-1.5736024020764605) q[3];
rz(2.9118671860419436) q[3];
ry(-2.6885898418531915) q[4];
rz(1.5688613418266284) q[4];
ry(9.917754543931068e-05) q[5];
rz(-2.386143747041936) q[5];
ry(-0.37237159090398025) q[6];
rz(0.780929663788424) q[6];
ry(0.0394911451879043) q[7];
rz(0.0031699140386161058) q[7];
ry(2.4727512573328605) q[8];
rz(-0.08968866099332252) q[8];
ry(-0.5781974771374551) q[9];
rz(2.2137972719170733) q[9];
ry(1.570819006322115) q[10];
rz(-1.570887771891262) q[10];
ry(0.9703201008009428) q[11];
rz(-3.009710658946123) q[11];
ry(-1.5708330793816596) q[12];
rz(-1.5708268935668102) q[12];
ry(2.5206081704884755) q[13];
rz(-1.5581491809292496) q[13];
ry(-1.571807594398889) q[14];
rz(1.6941553667509046) q[14];
ry(-0.2541770128841491) q[15];
rz(-1.6348963558953535) q[15];
ry(-1.571057676719585) q[16];
rz(1.57204682008458) q[16];
ry(-1.0910513844986927) q[17];
rz(1.8010435605775141) q[17];
ry(-2.075090500520089) q[18];
rz(-0.017740391383759047) q[18];
ry(-3.1307799813220587) q[19];
rz(-0.7425583535742744) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.111557115938013) q[0];
rz(1.2059865123252005) q[0];
ry(2.9492386411991545) q[1];
rz(1.1034639311125112) q[1];
ry(-1.3197198615356553) q[2];
rz(3.141049685245647) q[2];
ry(-1.570927563865098) q[3];
rz(0.7960754129413177) q[3];
ry(1.5708173763413287) q[4];
rz(-1.5762355776650738) q[4];
ry(-1.5702458216219235) q[5];
rz(2.0346612368120933) q[5];
ry(1.8410166206440037) q[6];
rz(1.767164293531784) q[6];
ry(0.046812299295994904) q[7];
rz(1.414046970853317) q[7];
ry(0.00018454015984165108) q[8];
rz(-2.955558162062995) q[8];
ry(0.00033156545703125955) q[9];
rz(0.9255281450578464) q[9];
ry(-1.4530286926957843) q[10];
rz(3.1329398017467334) q[10];
ry(0.2828609968325176) q[11];
rz(0.00031940557687624543) q[11];
ry(-1.5833938848497928) q[12];
rz(3.1415243542370903) q[12];
ry(1.573845431662579) q[13];
rz(0.0451227932180599) q[13];
ry(0.007770555124970384) q[14];
rz(-0.12337603448490242) q[14];
ry(-0.5696566084042866) q[15];
rz(1.5841546106729885) q[15];
ry(2.6986681030261104) q[16];
rz(0.29165743181725196) q[16];
ry(0.0011380045528435192) q[17];
rz(0.5320806534424714) q[17];
ry(1.5879352571602015) q[18];
rz(-1.0282289971521363) q[18];
ry(0.007932769574304246) q[19];
rz(0.5156583943695374) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.3043824078701425) q[0];
rz(-0.48128614235448014) q[0];
ry(-7.468436158930558e-05) q[1];
rz(-1.176861362294499) q[1];
ry(1.5360728479411867) q[2];
rz(-3.13757890924648) q[2];
ry(3.1316003614662153) q[3];
rz(2.7982183721322214) q[3];
ry(-0.4264069239538125) q[4];
rz(0.004575079051337917) q[4];
ry(-1.5700765125964973) q[5];
rz(-7.334011541537677e-05) q[5];
ry(-3.132785664899754) q[6];
rz(3.0794051371968534) q[6];
ry(-0.4188792149850855) q[7];
rz(1.557149264816518) q[7];
ry(0.9048672585113984) q[8];
rz(1.5111316771746703) q[8];
ry(1.4926405021889178) q[9];
rz(-1.6375316615462685) q[9];
ry(0.00011567943792461358) q[10];
rz(-3.132784299622494) q[10];
ry(-0.9591992591747736) q[11];
rz(3.1412181288502277) q[11];
ry(-1.2636992265021392) q[12];
rz(-3.141427509229626) q[12];
ry(-1.1577078327909327) q[13];
rz(3.0817604811484056) q[13];
ry(-1.6719144541752737) q[14];
rz(-0.0312208582268223) q[14];
ry(-1.9354750113300785) q[15];
rz(-3.101132124963723) q[15];
ry(-0.0005065337643461731) q[16];
rz(2.852234768684595) q[16];
ry(-1.6121023875576002) q[17];
rz(3.048338691625444) q[17];
ry(-0.37380510445556414) q[18];
rz(1.9496382393434661) q[18];
ry(-1.5691501797499459) q[19];
rz(2.106654513549851) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.1505705373810594) q[0];
rz(2.772918365449858) q[0];
ry(-2.302541811979694) q[1];
rz(-1.567681334592382) q[1];
ry(-1.0900137519995696) q[2];
rz(-1.5727789550203983) q[2];
ry(-3.1361061351696913) q[3];
rz(0.3427532746078513) q[3];
ry(2.195012429567147) q[4];
rz(-0.28342233020777474) q[4];
ry(1.5691438220639355) q[5];
rz(-0.7822962136254116) q[5];
ry(-1.5984266070601905) q[6];
rz(-3.141577081204073) q[6];
ry(-0.44792628179921135) q[7];
rz(-1.5880193294139282) q[7];
ry(1.5378591486160766) q[8];
rz(2.153220453236245e-05) q[8];
ry(0.47075569403266576) q[9];
rz(-3.1285107856225554) q[9];
ry(1.0196588437488412) q[10];
rz(-0.00016254590862718746) q[10];
ry(-2.155862625503765) q[11];
rz(-1.7808259565228874) q[11];
ry(-1.5779717044638442) q[12];
rz(-3.1415571374939857) q[12];
ry(1.567530791090468) q[13];
rz(0.0006126818432356701) q[13];
ry(-0.0003128806620207314) q[14];
rz(0.031240884129195127) q[14];
ry(-1.5724499322972854) q[15];
rz(-0.36627583910117906) q[15];
ry(3.0248531236152796) q[16];
rz(-1.5705570845323074) q[16];
ry(-1.5695705557273074) q[17];
rz(1.5715807041670624) q[17];
ry(-1.5612133624028905) q[18];
rz(-1.3137391041992548) q[18];
ry(2.7423980217759003) q[19];
rz(-2.8618741109083006) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.00424107211969904) q[0];
rz(0.8263064785650154) q[0];
ry(-1.5715349316030558) q[1];
rz(-1.56697043111703) q[1];
ry(1.5703488088986735) q[2];
rz(-1.5013513429832641) q[2];
ry(0.010679907968530244) q[3];
rz(-3.052803570684887) q[3];
ry(0.00040002946288724306) q[4];
rz(1.8566241446738476) q[4];
ry(3.141411675466319) q[5];
rz(0.6882385045495517) q[5];
ry(1.5708937975259296) q[6];
rz(1.5710191669461144) q[6];
ry(2.8360665358368586) q[7];
rz(3.582304579730078e-05) q[7];
ry(1.5707525168985548) q[8];
rz(1.5708668911076746) q[8];
ry(0.0006237342097774601) q[9];
rz(1.5734357554164486) q[9];
ry(-1.5708663421072395) q[10];
rz(1.570791676018235) q[10];
ry(0.02145576573935577) q[11];
rz(0.21012403608778385) q[11];
ry(1.571823339585248) q[12];
rz(1.5707495352302194) q[12];
ry(-1.56871230831638) q[13];
rz(1.5148195519465713) q[13];
ry(-1.748169979438884) q[14];
rz(1.570848996029349) q[14];
ry(1.7167815698760345) q[15];
rz(0.3434600360885928) q[15];
ry(2.804901288404827) q[16];
rz(-1.5707310395821885) q[16];
ry(-1.5709065963079345) q[17];
rz(-1.3155967213867261) q[17];
ry(3.1415182836351607) q[18];
rz(-0.7967695200310511) q[18];
ry(1.5513895386257925) q[19];
rz(1.8680296593001602) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.5707159193577578) q[0];
rz(0.31645529531188155) q[0];
ry(1.6883830967223803) q[1];
rz(-3.0281919293602235) q[1];
ry(-1.5707923900331928) q[2];
rz(-1.2541171489634217) q[2];
ry(-1.570876093202195) q[3];
rz(1.6679969854210104) q[3];
ry(1.574740434782353) q[4];
rz(-2.200688837422864) q[4];
ry(1.5120946612238306) q[5];
rz(-1.9202667853895408) q[5];
ry(-1.5708094161722528) q[6];
rz(1.9168922054808002) q[6];
ry(-1.5709598077126916) q[7];
rz(3.117825223789448) q[7];
ry(1.5708346563962723) q[8];
rz(-1.2629356348577314) q[8];
ry(-1.68997637695009) q[9];
rz(0.4753157512421174) q[9];
ry(1.5708580292471472) q[10];
rz(-0.4008594341105384) q[10];
ry(1.5708630539793889) q[11];
rz(2.501946797727687) q[11];
ry(-1.5708775948596243) q[12];
rz(1.0672372146694888) q[12];
ry(1.4912187377721722) q[13];
rz(-2.6328182372301523) q[13];
ry(1.570805625423599) q[14];
rz(-2.844831486524457) q[14];
ry(-3.141060058754429) q[15];
rz(-0.9812622409307679) q[15];
ry(-1.570753400719531) q[16];
rz(-2.844434032993603) q[16];
ry(-1.5714060439506123) q[17];
rz(1.7561730416249868) q[17];
ry(0.0001362840854176536) q[18];
rz(-0.22831810459789195) q[18];
ry(1.959066604816595) q[19];
rz(-2.156022957420179) q[19];