OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.7539671798198029) q[0];
ry(-1.6428318092666065) q[1];
cx q[0],q[1];
ry(-0.6478840176905091) q[0];
ry(1.5080709090849682) q[1];
cx q[0],q[1];
ry(3.0818882129540355) q[1];
ry(-0.22531691007878152) q[2];
cx q[1],q[2];
ry(0.04650646747064524) q[1];
ry(3.116726781678815) q[2];
cx q[1],q[2];
ry(-0.4509798968630107) q[2];
ry(-2.0277729992898426) q[3];
cx q[2],q[3];
ry(3.0815898996274256) q[2];
ry(2.6896792761652577) q[3];
cx q[2],q[3];
ry(-0.6577008751296087) q[0];
ry(2.1806799699507824) q[1];
cx q[0],q[1];
ry(0.4043579746881916) q[0];
ry(-1.3450289802948356) q[1];
cx q[0],q[1];
ry(3.075656280004665) q[1];
ry(-0.9526463591369403) q[2];
cx q[1],q[2];
ry(-1.6713517491163947) q[1];
ry(1.8490363709789348) q[2];
cx q[1],q[2];
ry(1.7521173675963753) q[2];
ry(-1.2401232790961874) q[3];
cx q[2],q[3];
ry(2.2072480908074876) q[2];
ry(-2.9213166189968556) q[3];
cx q[2],q[3];
ry(-1.0621862694469204) q[0];
ry(1.042367570313148) q[1];
cx q[0],q[1];
ry(-2.517403496977207) q[0];
ry(1.4169744226884988) q[1];
cx q[0],q[1];
ry(-2.3147212787298383) q[1];
ry(-1.0758024583611796) q[2];
cx q[1],q[2];
ry(0.34106770567219347) q[1];
ry(-2.7399107328318704) q[2];
cx q[1],q[2];
ry(2.6423244880793115) q[2];
ry(1.7227536307380884) q[3];
cx q[2],q[3];
ry(1.4168453849792302) q[2];
ry(-2.1363647623472044) q[3];
cx q[2],q[3];
ry(0.8953515634106849) q[0];
ry(-2.99348442302713) q[1];
cx q[0],q[1];
ry(-2.1421074806291216) q[0];
ry(-2.526191856986802) q[1];
cx q[0],q[1];
ry(-1.171193395489441) q[1];
ry(2.405328604844299) q[2];
cx q[1],q[2];
ry(2.968433382688896) q[1];
ry(0.6063813553829615) q[2];
cx q[1],q[2];
ry(1.2749186535951342) q[2];
ry(-2.7269180481288404) q[3];
cx q[2],q[3];
ry(-1.6617337504373637) q[2];
ry(0.04612627023995142) q[3];
cx q[2],q[3];
ry(-2.985014971280884) q[0];
ry(0.30721314003485123) q[1];
cx q[0],q[1];
ry(-2.6154327511297555) q[0];
ry(0.9346065276678983) q[1];
cx q[0],q[1];
ry(-1.5047611762200024) q[1];
ry(2.6009296349957802) q[2];
cx q[1],q[2];
ry(-2.219179096636428) q[1];
ry(-0.3281918326043618) q[2];
cx q[1],q[2];
ry(2.268651786007205) q[2];
ry(1.6747543416379214) q[3];
cx q[2],q[3];
ry(-2.599314973999807) q[2];
ry(-2.3993577490725504) q[3];
cx q[2],q[3];
ry(2.669857454185243) q[0];
ry(-1.973976858110503) q[1];
cx q[0],q[1];
ry(-1.4459692426245714) q[0];
ry(0.4620329027834425) q[1];
cx q[0],q[1];
ry(2.62325456153917) q[1];
ry(-2.8163398951341687) q[2];
cx q[1],q[2];
ry(2.765733746366841) q[1];
ry(-0.753154671560993) q[2];
cx q[1],q[2];
ry(1.6015307148445386) q[2];
ry(-2.245933544229092) q[3];
cx q[2],q[3];
ry(0.3881378770667761) q[2];
ry(-1.8150415133811855) q[3];
cx q[2],q[3];
ry(-2.05864749457451) q[0];
ry(0.9091454548317648) q[1];
cx q[0],q[1];
ry(-1.9106154002711397) q[0];
ry(2.0384791891545886) q[1];
cx q[0],q[1];
ry(-2.153689027701975) q[1];
ry(2.6842419782176696) q[2];
cx q[1],q[2];
ry(-2.1294066421720426) q[1];
ry(0.8095418167368953) q[2];
cx q[1],q[2];
ry(1.3253987742833022) q[2];
ry(1.9186446195577405) q[3];
cx q[2],q[3];
ry(2.876586156104863) q[2];
ry(-1.641243325611596) q[3];
cx q[2],q[3];
ry(0.9587314786951745) q[0];
ry(2.914370255482299) q[1];
cx q[0],q[1];
ry(-2.839730300043174) q[0];
ry(1.2541702287944805) q[1];
cx q[0],q[1];
ry(2.3727675384256814) q[1];
ry(0.08103782616814659) q[2];
cx q[1],q[2];
ry(1.8788878908317301) q[1];
ry(-0.003921366351683359) q[2];
cx q[1],q[2];
ry(0.39834011869029506) q[2];
ry(1.1960966538335054) q[3];
cx q[2],q[3];
ry(-0.7141004447753152) q[2];
ry(2.0902633903671237) q[3];
cx q[2],q[3];
ry(2.0900706760589736) q[0];
ry(0.8753749282941445) q[1];
ry(0.1524762525614045) q[2];
ry(-0.10414167817670056) q[3];