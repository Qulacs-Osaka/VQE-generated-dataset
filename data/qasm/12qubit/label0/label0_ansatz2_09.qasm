OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.469934678459927e-06) q[0];
rz(-1.3925308171522701) q[0];
ry(0.000349567500888881) q[1];
rz(-1.7482645375057033) q[1];
ry(-0.00015647496923446418) q[2];
rz(1.2623883774979203) q[2];
ry(-0.00017285196950323785) q[3];
rz(-1.7254939988132767) q[3];
ry(2.901230615804892) q[4];
rz(1.3939440881198477) q[4];
ry(1.5640896438461873) q[5];
rz(1.679231421832442) q[5];
ry(3.1367235726840517) q[6];
rz(1.783128786033922) q[6];
ry(-3.107538196180715) q[7];
rz(-0.00836662006065437) q[7];
ry(-7.313538997788173e-05) q[8];
rz(2.058297029122348) q[8];
ry(0.00014694162925277252) q[9];
rz(1.3105738188549902) q[9];
ry(5.9501541871398445e-06) q[10];
rz(0.21816665239696323) q[10];
ry(-3.1415712077624214) q[11];
rz(2.8135912805637022) q[11];
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
ry(-1.5708112555029021) q[0];
rz(0.6119298925699439) q[0];
ry(1.87850656369055) q[1];
rz(0.00015157748312244482) q[1];
ry(0.0007461188953139876) q[2];
rz(-0.7017069337241937) q[2];
ry(3.1256231932194574) q[3];
rz(1.0370122886241724) q[3];
ry(3.0007224553934124) q[4];
rz(1.9856506108096423) q[4];
ry(-0.19654980664193736) q[5];
rz(1.6649675865521631) q[5];
ry(0.022100106136495542) q[6];
rz(-1.3668394937010853) q[6];
ry(-1.5732998627608696) q[7];
rz(3.136572956895302) q[7];
ry(-1.575989596372952) q[8];
rz(-2.023143644795012) q[8];
ry(1.57215158358294) q[9];
rz(2.596423705421704) q[9];
ry(3.14135036285517) q[10];
rz(-3.136879957705462) q[10];
ry(1.5909286977551238) q[11];
rz(0.3899028879458669) q[11];
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
ry(3.1415293049251565) q[0];
rz(-0.9588643361911408) q[0];
ry(1.5780233431411126) q[1];
rz(-3.118197049132835) q[1];
ry(-0.9255049291999703) q[2];
rz(-3.1409827645452557) q[2];
ry(0.33188730582567716) q[3];
rz(-3.1142878032611634) q[3];
ry(0.0003312432463572107) q[4];
rz(-0.4136104048220308) q[4];
ry(-0.003268714853182253) q[5];
rz(-1.0349540439961542) q[5];
ry(0.00020272818316335875) q[6];
rz(-1.6236994238875728) q[6];
ry(0.013566441847693872) q[7];
rz(-3.0151512368814086) q[7];
ry(0.0007929376634079333) q[8];
rz(-2.4609383191935357) q[8];
ry(1.970344898527543e-05) q[9];
rz(-0.8238009534606812) q[9];
ry(-1.6215673128475254e-05) q[10];
rz(-2.6246149173763675) q[10];
ry(3.1413968838517268) q[11];
rz(2.6532403285248964) q[11];
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
ry(1.5676112583202784) q[0];
rz(-1.5707950959433914) q[0];
ry(-2.922878545439245) q[1];
rz(1.3778550810186645) q[1];
ry(-2.300753942750061) q[2];
rz(-1.8713683434083768) q[2];
ry(-0.044976002865578124) q[3];
rz(0.6718520790443705) q[3];
ry(-1.5721645727486955) q[4];
rz(0.016642367536632285) q[4];
ry(0.0001926152678611297) q[5];
rz(0.5652613615914834) q[5];
ry(0.0007647809892939439) q[6];
rz(0.26797868607158026) q[6];
ry(-0.00021020289957961668) q[7];
rz(0.12887951757006003) q[7];
ry(3.1349470616697963) q[8];
rz(-2.1324998289397508) q[8];
ry(3.1398856615725457) q[9];
rz(3.050278267834803) q[9];
ry(-1.2599914434474611e-05) q[10];
rz(-1.857745154412048) q[10];
ry(0.0002711941193557622) q[11];
rz(1.4775837286545712) q[11];
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
ry(1.5846554077176143) q[0];
rz(-1.5707290911442948) q[0];
ry(-0.0002956342029509784) q[1];
rz(-2.92558339673401) q[1];
ry(3.1411026278499428) q[2];
rz(-2.9919942555844714) q[2];
ry(0.0006238506398418861) q[3];
rz(0.5720231762892807) q[3];
ry(1.526971966598683) q[4];
rz(3.1398778506557132) q[4];
ry(-1.5527439674711385) q[5];
rz(1.5755678259636778) q[5];
ry(1.5699831969033633) q[6];
rz(1.5378085614633674) q[6];
ry(-1.5698195236748964) q[7];
rz(-1.5736734144998086) q[7];
ry(-3.124784652083562) q[8];
rz(-1.385684451114905) q[8];
ry(-3.1388209524553266) q[9];
rz(-0.5812853223591831) q[9];
ry(0.0002386679131086462) q[10];
rz(-0.11345752916016671) q[10];
ry(-0.000734364868532716) q[11];
rz(0.5135565502250569) q[11];
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
ry(-1.8054122441105347) q[0];
rz(-0.5738763931141436) q[0];
ry(-1.1961589530062389) q[1];
rz(-1.5710343716969486) q[1];
ry(0.0011764942660627562) q[2];
rz(1.786153042509335) q[2];
ry(-0.0006263035532816374) q[3];
rz(-0.5783384567911641) q[3];
ry(1.561073703555964) q[4];
rz(1.6460646514448045) q[4];
ry(-1.5910672871042397) q[5];
rz(1.7233318842641077) q[5];
ry(-1.5737577547610342) q[6];
rz(0.5548456553207235) q[6];
ry(-1.621338474367244) q[7];
rz(0.8861454764719421) q[7];
ry(-3.140929100918883) q[8];
rz(-2.5993001435717495) q[8];
ry(-0.00807987554257572) q[9];
rz(-1.011488007005151) q[9];
ry(-5.925289724650895e-05) q[10];
rz(2.215414902172875) q[10];
ry(-3.1390359022598786) q[11];
rz(-2.026957375466334) q[11];
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
ry(3.1415895489008134) q[0];
rz(2.5676305763143388) q[0];
ry(-1.5738540968835228) q[1];
rz(-3.1109709009362008) q[1];
ry(-3.1412890366547326) q[2];
rz(-0.9602846952137885) q[2];
ry(3.1407406890049328) q[3];
rz(1.926742740044192) q[3];
ry(3.1413503605712196) q[4];
rz(1.8290062730003211) q[4];
ry(-0.0011583220142576088) q[5];
rz(-1.9065665156064293) q[5];
ry(0.0006019257174836258) q[6];
rz(0.5490487966492612) q[6];
ry(-0.0062104903693605) q[7];
rz(-0.24018875961526948) q[7];
ry(-0.9817128858379442) q[8];
rz(3.116844146362884) q[8];
ry(-1.4274696983884008) q[9];
rz(-1.7070565328402316) q[9];
ry(-3.141159118721268) q[10];
rz(2.173355651638492) q[10];
ry(1.3565568958755672) q[11];
rz(-1.5300578425421025) q[11];
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
ry(2.4262849384941383) q[0];
rz(1.5709502277462484) q[0];
ry(-0.0003516113811432575) q[1];
rz(2.004734571277864) q[1];
ry(3.1415372367457763) q[2];
rz(3.082757361826207) q[2];
ry(0.00012440236483346467) q[3];
rz(0.6821906247095624) q[3];
ry(0.00039705237661813447) q[4];
rz(-1.640628231553538) q[4];
ry(0.0036938926946477896) q[5];
rz(0.2510269505577636) q[5];
ry(3.129214327613288) q[6];
rz(-1.7002489125916052) q[6];
ry(-3.071522858618863) q[7];
rz(0.5614718667089722) q[7];
ry(1.406977099287161) q[8];
rz(-2.7669697261491013) q[8];
ry(-2.0315801623282246) q[9];
rz(2.376396884233774) q[9];
ry(3.1413590388612382) q[10];
rz(1.196251222098038) q[10];
ry(2.3970363858327635) q[11];
rz(0.89514098666535) q[11];
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
ry(2.1804527746470086) q[0];
rz(-1.896370695552808) q[0];
ry(-3.140901893988998) q[1];
rz(2.8091213227003307) q[1];
ry(3.140645932934656) q[2];
rz(0.3792916849413553) q[2];
ry(0.0005488526788469327) q[3];
rz(1.046846065854763) q[3];
ry(0.0033798394273718557) q[4];
rz(1.9034127481784129) q[4];
ry(3.1226003230428896) q[5];
rz(-2.8209935385088487) q[5];
ry(3.1341213198655087) q[6];
rz(0.8801783680584226) q[6];
ry(0.4829088869184962) q[7];
rz(-2.5292848223884374) q[7];
ry(-0.937605456652218) q[8];
rz(-0.13924437582743998) q[8];
ry(2.966171527238796) q[9];
rz(-0.716514352919071) q[9];
ry(-3.1411150720849412) q[10];
rz(-1.8128502730073839) q[10];
ry(0.09413861433592441) q[11];
rz(-0.8328599181126671) q[11];
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
ry(3.141337700765073) q[0];
rz(1.2446379645444132) q[0];
ry(3.14115840234433) q[1];
rz(0.7760504199941377) q[1];
ry(3.14120945321145) q[2];
rz(0.3347900460292869) q[2];
ry(-0.0004658394139784151) q[3];
rz(0.5200221239522356) q[3];
ry(0.007761953934144472) q[4];
rz(1.1269703042602965) q[4];
ry(3.134348970692038) q[5];
rz(1.8205227350524273) q[5];
ry(0.02471906549543501) q[6];
rz(-0.5236490029846922) q[6];
ry(0.003662613093911027) q[7];
rz(-2.0531473592720264) q[7];
ry(-1.5634033790139164) q[8];
rz(1.4502825043804155) q[8];
ry(1.5822848217605383) q[9];
rz(-1.8637680091820732) q[9];
ry(-1.193837494017913e-05) q[10];
rz(-1.3063137752164968) q[10];
ry(-3.1114467746178334) q[11];
rz(1.325428937463494) q[11];
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
ry(0.3682820394280056) q[0];
rz(1.5712429147703293) q[0];
ry(-0.2567478276838526) q[1];
rz(-2.633053485558246) q[1];
ry(0.0015847797444871812) q[2];
rz(2.949146672079003) q[2];
ry(3.140362867790629) q[3];
rz(1.2614986830534765) q[3];
ry(-1.5813609276595872) q[4];
rz(0.36831654320982204) q[4];
ry(1.5859300625971917) q[5];
rz(0.2501024963302552) q[5];
ry(-1.5501697670134025) q[6];
rz(-2.8662528545141295) q[6];
ry(-1.589015145058288) q[7];
rz(-1.2479707169695855) q[7];
ry(-1.5248351591468996) q[8];
rz(0.8451553173342422) q[8];
ry(-0.9982140270636313) q[9];
rz(1.711763760512035) q[9];
ry(1.5699690369395043) q[10];
rz(3.141522244242938) q[10];
ry(-1.292741733326866) q[11];
rz(-1.8177858695187967) q[11];
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
ry(1.5709092372682754) q[0];
rz(1.3524271390695013) q[0];
ry(-5.7759502258099335e-05) q[1];
rz(-0.642434470154411) q[1];
ry(-3.141571386449298) q[2];
rz(-1.811682246903013) q[2];
ry(3.1415745458904856) q[3];
rz(0.9203405516279747) q[3];
ry(3.1415781190559624) q[4];
rz(3.0676412532404655) q[4];
ry(1.8677583022263186e-05) q[5];
rz(-1.6608624900029694) q[5];
ry(-3.1415689814934247) q[6];
rz(0.26962778215422395) q[6];
ry(3.141568169931785) q[7];
rz(-0.8805401255172728) q[7];
ry(2.5682232340360454e-05) q[8];
rz(-3.0950292083687376) q[8];
ry(0.00038061302731051683) q[9];
rz(-2.944874711787989) q[9];
ry(1.570500491945863) q[10];
rz(-0.003948341820131596) q[10];
ry(-3.14152070160435) q[11];
rz(1.2151634339640331) q[11];
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
ry(-0.0005086764733171099) q[0];
rz(1.7891601780319677) q[0];
ry(-3.1405959454283954) q[1];
rz(-1.699970028665603) q[1];
ry(-1.5705822973649681) q[2];
rz(-3.1409231650292786) q[2];
ry(-1.5850702207148337) q[3];
rz(0.008599886099770932) q[3];
ry(-0.010160065299881893) q[4];
rz(-1.128100122798938) q[4];
ry(0.00016743389252576345) q[5];
rz(-0.17198737054099486) q[5];
ry(1.5133143395438813) q[6];
rz(1.5892990393383322) q[6];
ry(-3.1203626026646454) q[7];
rz(1.5291344913592813) q[7];
ry(0.006052411559188476) q[8];
rz(1.120888134586307) q[8];
ry(0.003163396765916815) q[9];
rz(2.899326057131004) q[9];
ry(-1.5659286382003996) q[10];
rz(1.3013792449145019) q[10];
ry(0.058238215594382775) q[11];
rz(-1.585694317218729) q[11];