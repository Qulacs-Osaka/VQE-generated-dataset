OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.5424724748501557) q[0];
rz(-3.000143495105296) q[0];
ry(-0.5991416194692123) q[1];
rz(-2.336116586025697) q[1];
ry(1.2194463980379018) q[2];
rz(-0.2941183881563197) q[2];
ry(2.912727199865895) q[3];
rz(-0.7475337241767317) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.9557334907095467) q[0];
rz(-0.33491025888824305) q[0];
ry(2.488663005647262) q[1];
rz(0.7450855621119126) q[1];
ry(2.176713820757448) q[2];
rz(-0.18002241499702173) q[2];
ry(0.1317196787955135) q[3];
rz(-2.0038882513947334) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.9103573071843832) q[0];
rz(-2.058709849478994) q[0];
ry(-3.0526036301531945) q[1];
rz(0.6811173323735326) q[1];
ry(0.3996161366277482) q[2];
rz(1.6722413193686598) q[2];
ry(1.364074128515097) q[3];
rz(-2.146829918668505) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.9104895611744137) q[0];
rz(1.6029009763465436) q[0];
ry(2.742878085131213) q[1];
rz(-2.7347179545779547) q[1];
ry(-1.5334187662879608) q[2];
rz(-1.6708489681054477) q[2];
ry(1.3915682900129653) q[3];
rz(-0.48644527721313663) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.1090397453338632) q[0];
rz(3.0580391686055886) q[0];
ry(2.2133825090589028) q[1];
rz(1.5089667813462835) q[1];
ry(-2.314774456020506) q[2];
rz(1.5010188373007356) q[2];
ry(0.5813790997933257) q[3];
rz(-1.0899509696304164) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.822029994964958) q[0];
rz(2.064231869947555) q[0];
ry(0.3520940678482978) q[1];
rz(-1.4855737804124498) q[1];
ry(1.666224430640974) q[2];
rz(0.259076999726493) q[2];
ry(-2.2628826682794383) q[3];
rz(0.9744858866728571) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.7463025406066834) q[0];
rz(2.6996674829635996) q[0];
ry(3.0370779429683847) q[1];
rz(-1.2744612779068758) q[1];
ry(0.9306080282608747) q[2];
rz(2.2793085335684435) q[2];
ry(0.6839856749027072) q[3];
rz(-2.034034608575495) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.25888474757322655) q[0];
rz(2.643784711375226) q[0];
ry(-2.8335645329480315) q[1];
rz(-2.1552465840810835) q[1];
ry(1.7981060859383717) q[2];
rz(3.0783880908631422) q[2];
ry(0.0893895610719655) q[3];
rz(2.2221706739411937) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.8443856523505198) q[0];
rz(-3.0563249207713787) q[0];
ry(-1.8754695897172744) q[1];
rz(-3.1389630448957666) q[1];
ry(-0.5868770376698952) q[2];
rz(1.6256374565673655) q[2];
ry(2.0170664660216033) q[3];
rz(-0.7687693146243951) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.3186615289924586) q[0];
rz(-2.675712853104185) q[0];
ry(1.7883901865564207) q[1];
rz(0.7180188341106648) q[1];
ry(-0.3385507567816272) q[2];
rz(-0.8415867450340038) q[2];
ry(2.7637786517251683) q[3];
rz(-0.2363032927147284) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.8652801927590374) q[0];
rz(-2.130945120663289) q[0];
ry(1.219379390506639) q[1];
rz(2.249588905843165) q[1];
ry(0.06441648983127113) q[2];
rz(-2.5854215772752873) q[2];
ry(-2.5805402123472305) q[3];
rz(1.5683492114132396) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.16267912871922352) q[0];
rz(-1.6891162227964998) q[0];
ry(0.2292518565285091) q[1];
rz(-0.8890194140381513) q[1];
ry(2.3391617641643903) q[2];
rz(-2.8936523227678395) q[2];
ry(0.36547502517611985) q[3];
rz(0.052361589188352646) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.8165497478561488) q[0];
rz(-1.4239966571322895) q[0];
ry(0.7430242104280698) q[1];
rz(-2.325084715474094) q[1];
ry(-2.113886568657222) q[2];
rz(3.1411646424038966) q[2];
ry(-1.9370562362210186) q[3];
rz(-0.7936143493962612) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.034096530273678) q[0];
rz(2.773071768832736) q[0];
ry(2.6876578507499786) q[1];
rz(-0.3490547890045157) q[1];
ry(-2.5698608317140166) q[2];
rz(-1.7739691787452145) q[2];
ry(-2.818327874501072) q[3];
rz(-1.2765267900661712) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.700148658250865) q[0];
rz(2.163683050394721) q[0];
ry(1.647850287235343) q[1];
rz(-1.4595762465708013) q[1];
ry(2.730816298267245) q[2];
rz(-1.0189262212170354) q[2];
ry(2.139449318800885) q[3];
rz(-3.0397566526792126) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.5993338632251874) q[0];
rz(1.161992227558933) q[0];
ry(1.6710638769538868) q[1];
rz(-0.2073156254876727) q[1];
ry(-3.045337162260955) q[2];
rz(2.0634497593814674) q[2];
ry(-1.1200317786535416) q[3];
rz(-3.0908080938799882) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.2243331597671272) q[0];
rz(1.4621755778987613) q[0];
ry(-1.8788514612327525) q[1];
rz(2.391082020754322) q[1];
ry(0.7654719280354244) q[2];
rz(-2.004535799823603) q[2];
ry(0.625643746874414) q[3];
rz(-0.05126689943107846) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.4427836332986081) q[0];
rz(1.1614941806280368) q[0];
ry(0.7047894205702134) q[1];
rz(-2.901154436438347) q[1];
ry(-1.8019303280914603) q[2];
rz(1.7095839028697377) q[2];
ry(0.6735380138440386) q[3];
rz(-3.0931896264047087) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.9804162836080192) q[0];
rz(-1.0553137683795484) q[0];
ry(-2.511879970008306) q[1];
rz(-2.90915870784854) q[1];
ry(-2.191658971657038) q[2];
rz(0.17546687661989416) q[2];
ry(-0.6091020352499324) q[3];
rz(-1.255950227544078) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.13967827586006) q[0];
rz(2.3527386541857975) q[0];
ry(0.28588879611162726) q[1];
rz(-1.6667334661154936) q[1];
ry(2.1850607829277573) q[2];
rz(-1.057328457464802) q[2];
ry(-1.5347132167860684) q[3];
rz(-0.024906438857946963) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.4117223494342264) q[0];
rz(-2.2312775973332384) q[0];
ry(1.3009572171081327) q[1];
rz(-2.821243105355045) q[1];
ry(2.0539044164849853) q[2];
rz(-2.153852671659531) q[2];
ry(-2.942279671148265) q[3];
rz(-0.7174642580344303) q[3];